"""Bilevel optimization trainer for learned graph rewiring.

Three differentiation strategies for the outer (policy) gradient:

  1. first_order  (DARTS-like)
     Single inner step with detached graph, then outer step on val loss.

  2. unrolled
     K inner steps retaining the computation graph, outer gradient backprops
     through the full unrolled inner optimization trajectory.

  3. implicit
     Approximate the implicit gradient via Neumann series.

Formulation:
  phi* = argmin_{phi} L_val(theta*(phi), phi)
  s.t.  theta*(phi) = argmin_{theta} L_train(theta, phi)

Edge injection approach (following LDS-GNN / IDGL patterns):
  Instead of converting adj_dense to a new SparseTensor, we preserve the
  original batch SparseTensor's structure (row/col) and only replace edge
  values from the policy output.  This guarantees:
    - self-loop status identical to original batch
    - CSR sort order identical → GCNConv normalize behaves identically
    - GCNConv cached/normalize settings untouched
    - Loss computation delegated to tsl Predictor's predict_batch
"""

import math

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.utils.data import DataLoader
from lightning.pytorch.utilities import CombinedLoader
from tsl.data.loader import StaticGraphLoader
from tsl.engines import Predictor


def make_bilevel_loader(train_dl: DataLoader, val_dl: DataLoader) -> CombinedLoader:
    """Wrap train + val dataloaders into a CombinedLoader for bilevel training.

    Use as: trainer.fit(model, train_dataloaders=make_bilevel_loader(train_dl, val_dl))
    training_step receives batch = {"train": train_batch, "val": val_batch}.
    max_size_cycle ensures val is cycled if shorter than train.
    """
    return CombinedLoader({"train": train_dl, "val": val_dl}, mode="max_size_cycle")


class VanillaCombinedPredictor(Predictor):
    """Vanilla Predictor that unwraps CombinedLoader batch.

    Uses the "train" key from CombinedLoader but delegates to vanilla
    Predictor.training_step. This isolates data volume effects from
    manual-optimization effects.
    """

    def training_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            batch = batch[0]
        return super().training_step(batch["train"], batch_idx)


class E2EReweightPredictor(Predictor):
    """Single-level ablation: policy params trained jointly with model params.

    Uses the exact same RelativeReweightPolicy as BilevelPredictor, but
    adds policy parameters to the main optimizer (no inner/outer split,
    no warmup, no validation-based graph optimization).
    """

    def __init__(self, model_class, model_kwargs, policy, adj_dense,
                 grad_clip=5.0, **kwargs):
        super().__init__(model_class=model_class,
                         model_kwargs=model_kwargs,
                         **kwargs)
        self.policy = policy
        self.register_buffer('adj_dense', adj_dense)
        self.grad_clip = grad_clip

    def configure_optimizers(self):
        all_params = list(self.model.parameters()) + list(self.policy.parameters())
        opt = self.optim_class(all_params, **self.optim_kwargs)
        if self.scheduler_class is not None:
            sched_kwargs = dict(self.scheduler_kwargs)
            monitor = sched_kwargs.pop('monitor', None)
            sched = self.scheduler_class(opt, **sched_kwargs)
            conf = {'scheduler': sched, 'interval': 'epoch'}
            if monitor:
                conf['monitor'] = monitor
            return {'optimizer': opt, 'lr_scheduler': conf}
        return opt

    def _get_rewired_adj(self):
        adj_s, adj_t, scores_s, scores_t = self.policy(self.adj_dense)
        return adj_s

    def _inject_rewired_edges(self, batch, adj_s):
        if hasattr(self.model, 'set_ext_adj'):
            self.model.set_ext_adj(adj_s)
            return
        for module in self.model.modules():
            if hasattr(module, '_support') and module._support is not None:
                module._support = None
        ref = batch.input.edge_index
        if hasattr(ref, 'sparse_sizes'):
            from torch_sparse import SparseTensor
            N = ref.sparse_sizes()[0]
            row, col = torch.nonzero(adj_s, as_tuple=True)
            if row.shape[0] == ref.nnz():
                row_o, col_o, _ = ref.coo()
                new_val = adj_s[row_o, col_o]
                batch.input.edge_index = SparseTensor(
                    row=row_o, col=col_o, value=new_val,
                    sparse_sizes=(N, N), is_sorted=True
                )
            else:
                val = adj_s[row, col]
                batch.input.edge_index = SparseTensor(
                    row=row, col=col, value=val,
                    sparse_sizes=(N, N)
                )
        elif hasattr(ref, 'shape') and ref.dim() == 2:
            row, col = torch.nonzero(adj_s, as_tuple=True)
            batch.input.edge_index = torch.stack([row, col])
            batch.input.edge_weight = adj_s[row, col]

    def training_step(self, batch, batch_idx):
        self.policy.train()
        adj_s = self._get_rewired_adj()
        self._inject_rewired_edges(batch, adj_s)
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.policy.eval()
        adj_s = self._get_rewired_adj()
        self._inject_rewired_edges(batch, adj_s.detach())
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.policy.eval()
        adj_s = self._get_rewired_adj()
        self._inject_rewired_edges(batch, adj_s.detach())
        return super().test_step(batch, batch_idx)

    @torch.no_grad()
    def get_rewired_adj(self):
        self.policy.eval()
        adj_s, adj_t, scores_s, scores_t = self.policy(self.adj_dense)
        return adj_s.cpu(), adj_t.cpu()


class E2EReweightCombinedPredictor(E2EReweightPredictor):
    """E2E reweight trained on train+val combined (via CombinedLoader).

    Ablation to test whether bilevel's advantage comes from the optimization
    separation or simply from having access to validation data.
    """
    automatic_optimization = False

    def configure_optimizers(self):
        all_params = list(self.model.parameters()) + list(self.policy.parameters())
        opt = self.optim_class(all_params, **self.optim_kwargs)
        if self.scheduler_class is not None:
            sched_kwargs = dict(self.scheduler_kwargs)
            sched_kwargs.pop('monitor', None)
            self._scheduler = self.scheduler_class(opt, **sched_kwargs)
        else:
            self._scheduler = None
        return [opt]

    def on_train_epoch_end(self):
        if self._scheduler is not None:
            self._scheduler.step()

    def _compute_loss_from_batch(self, batch):
        y_hat_loss = self.predict_batch(batch, preprocess=False,
                                        postprocess=not self.scale_target)
        y_loss = batch.y
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y_loss)
        return self.loss_fn(y_hat_loss, y_loss, batch.get('mask'))

    def training_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            batch = batch[0]
        train_batch = batch["train"].to(self.device)
        val_batch = batch["val"].to(self.device)

        opt = self.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        self.model.train()
        self.policy.train()
        adj_s = self._get_rewired_adj()

        self._inject_rewired_edges(train_batch, adj_s)
        train_loss = self._compute_loss_from_batch(train_batch)

        self._inject_rewired_edges(val_batch, adj_s)
        val_loss = self._compute_loss_from_batch(val_batch)

        combined_loss = 0.5 * (train_loss + val_loss)

        opt.zero_grad()
        self.manual_backward(combined_loss)
        nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.policy.parameters()),
            self.grad_clip)
        opt.step()

        self.log('train_loss', train_loss, prog_bar=True)
        self.log('val_loss_combined', val_loss, prog_bar=True)
        return combined_loss


class BilevelPredictor(Predictor):
    """Predictor with bilevel optimization for learned rewiring.

    Expects training batches as {"train": batch, "val": batch} via CombinedLoader.

    Edge injection: modifies batch.input.edge_index values in-place from
    the policy's output adjacency, preserving the original SparseTensor
    structure. All forward/loss/metric computation is delegated to the
    parent tsl Predictor's predict_batch.
    """
    automatic_optimization = False

    def __init__(self, model_class, model_kwargs, policy, adj_dense,
                 bilevel_method='first_order',
                 inner_lr=1e-3, outer_lr=1e-4,
                 inner_steps=1, unroll_steps=5,
                 neumann_steps=5, neumann_alpha=0.1,
                 grad_clip=5.0,
                 outer_warmup_epochs=0, outer_cosine=False, max_epochs=200,
                 rewire_every_k=1,
                 freeze_phi=False,
                 outer_loss='val',
                 inner_batch_mode='reuse',
                 **kwargs):
        super().__init__(model_class=model_class,
                         model_kwargs=model_kwargs,
                         **kwargs)
        if inner_batch_mode not in ('reuse', 'fresh'):
            raise ValueError(
                f"inner_batch_mode must be 'reuse' or 'fresh', got {inner_batch_mode!r}")
        self.policy = policy
        self.register_buffer('adj_dense', adj_dense)
        self.bilevel_method = bilevel_method
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.unroll_steps = unroll_steps
        self.neumann_steps = neumann_steps
        self.neumann_alpha = neumann_alpha
        self.grad_clip = grad_clip
        self.outer_warmup_epochs = outer_warmup_epochs
        self.outer_cosine = outer_cosine
        self._max_epochs = max_epochs
        self.rewire_every_k = rewire_every_k
        self.freeze_phi = freeze_phi
        self.outer_loss = outer_loss
        self.inner_batch_mode = inner_batch_mode
        self._inner_train_dataset = None
        self._inner_batch_size = None
        self._inner_seed = None
        self._fresh_iter = None
        if freeze_phi:
            self.policy.requires_grad_(False)

    def configure_inner_batch_source(self, dataset, batch_size, seed):
        """Attach the train split used when inner_batch_mode='fresh'."""
        self._inner_train_dataset = dataset
        self._inner_batch_size = int(batch_size)
        self._inner_seed = int(seed)

    def on_train_epoch_start(self):
        if self.inner_batch_mode != 'fresh':
            return
        if self._inner_train_dataset is None:
            raise RuntimeError(
                "inner_batch_mode='fresh' requires configure_inner_batch_source() "
                "before trainer.fit()")
        g = torch.Generator()
        g.manual_seed(self._inner_seed + self.current_epoch * 1_000_003)
        # Match SpatioTemporalDataModule.train_dataloader: shuffle=True, drop_last=True.
        # Dedicated Generator keeps the sequence reproducible and independent of
        # the CombinedLoader's own shuffle RNG.
        fresh_dl = StaticGraphLoader(
            self._inner_train_dataset,
            batch_size=self._inner_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            generator=g,
        )
        self._fresh_iter = iter(fresh_dl)

    def _next_fresh_batch(self):
        try:
            batch = next(self._fresh_iter)
        except StopIteration:
            g = torch.Generator()
            g.manual_seed(
                self._inner_seed + self.current_epoch * 1_000_003 + 17)
            fresh_dl = StaticGraphLoader(
                self._inner_train_dataset,
                batch_size=self._inner_batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
                generator=g,
            )
            self._fresh_iter = iter(fresh_dl)
            batch = next(self._fresh_iter)
        return batch.to(self.device)

    @staticmethod
    def _batch_fingerprint(batch):
        y = batch.y.detach().float().reshape(-1)
        return (
            tuple(batch.y.shape),
            float(y.sum().item()),
            float(y[0].item()),
            float(y[-1].item()),
            float(y[y.numel() // 2].item()),
        )

    def configure_optimizers(self):
        inner_opt = self.optim_class(
            self.model.parameters(),
            **self.optim_kwargs,
        )
        outer_opt = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.outer_lr,
        )
        if self.scheduler_class is not None:
            sched_kwargs = dict(self.scheduler_kwargs)
            sched_kwargs.pop('monitor', None)
            self._inner_scheduler = self.scheduler_class(inner_opt, **sched_kwargs)
        else:
            self._inner_scheduler = None
        return [inner_opt, outer_opt]

    # ------------------------------------------------------------------
    # Outer LR scheduler (manual, since automatic_optimization=False)
    # ------------------------------------------------------------------

    def _get_outer_lr(self, epoch):
        """Compute outer lr with optional linear warmup + cosine decay."""
        if epoch < self.outer_warmup_epochs:
            return self.outer_lr * (epoch + 1) / self.outer_warmup_epochs
        if not self.outer_cosine:
            return self.outer_lr
        t = epoch - self.outer_warmup_epochs
        t_max = max(1, self._max_epochs - self.outer_warmup_epochs)
        return 1e-6 + (self.outer_lr - 1e-6) * 0.5 * (1 + math.cos(math.pi * t / t_max))

    def on_train_epoch_end(self):
        if self._inner_scheduler is not None:
            self._inner_scheduler.step()
            self.log('inner_lr', self._inner_scheduler.get_last_lr()[0])
        if self.outer_warmup_epochs > 0 or self.outer_cosine:
            new_lr = self._get_outer_lr(self.current_epoch + 1)
            outer_opt = self.optimizers()[1]
            for pg in outer_opt.param_groups:
                pg['lr'] = new_lr
            self.log('outer_lr', new_lr)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _get_rewired_adj(self):
        adj_s, adj_t, scores_s, scores_t = self.policy(self.adj_dense)
        return adj_s, adj_t

    def _inject_rewired_edges(self, batch, adj_s):
        """Inject rewired adjacency into the model.

        For edge_index-based models (DiffConv, GCN): replaces SparseTensor values.
        For models with set_ext_adj (AGCRNBilevel): passes dense adj to model directly.
        """
        # AGCRN-like models: pass adj directly to the model
        if hasattr(self.model, 'set_ext_adj'):
            self.model.set_ext_adj(adj_s)
            return
        # Clear DiffConv support cache (stale after edge injection)
        for module in self.model.modules():
            if hasattr(module, '_support') and module._support is not None:
                module._support = None
        ref = batch.input.edge_index
        if hasattr(ref, 'sparse_sizes'):
            from torch_sparse import SparseTensor
            N = ref.sparse_sizes()[0]
            row, col = torch.nonzero(adj_s, as_tuple=True)
            if row.shape[0] == ref.nnz():
                # Same topology: fast injection preserving CSR order
                row_o, col_o, _ = ref.coo()
                new_val = adj_s[row_o, col_o]
                batch.input.edge_index = SparseTensor(
                    row=row_o, col=col_o, value=new_val,
                    sparse_sizes=(N, N), is_sorted=True
                )
            else:
                # Topology changed (policy added/removed edges): full rebuild
                val = adj_s[row, col]
                batch.input.edge_index = SparseTensor(
                    row=row, col=col, value=val,
                    sparse_sizes=(N, N)
                )
        elif hasattr(ref, 'shape') and ref.dim() == 2:
            row, col = torch.nonzero(adj_s, as_tuple=True)
            batch.input.edge_index = torch.stack([row, col])
            batch.input.edge_weight = adj_s[row, col]

    def _compute_loss_from_batch(self, batch):
        """Compute loss exactly like tsl Predictor.training_step.

        Delegates forward + scaling to predict_batch, then computes loss.
        This is the single source of truth for loss computation — no
        custom inverse_transform or scale_target logic needed.
        """
        y_hat_loss = self.predict_batch(batch, preprocess=False,
                                        postprocess=not self.scale_target)
        y_loss = batch.y
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y_loss)
        return self.loss_fn(y_hat_loss, y_loss, batch.get('mask'))

    def _compute_loss_functional(self, batch, params):
        """Compute loss using functional_call (for unrolled differentiation).

        Same scaling logic as predict_batch, but uses a detached parameter
        dict to allow differentiating through the inner optimization.
        Assumes _inject_rewired_edges has already been called on batch.
        """
        inputs = {k: getattr(batch.input, k) for k in batch.input.keys()}
        if self.filter_forward_kwargs:
            inputs = self._filter_forward_kwargs(inputs)
        y_hat = functional_call(self.model, params, kwargs=inputs)
        y_loss = batch.y
        if not self.scale_target:
            transform = batch.get('transform')
            trans = transform.get('y') if transform else None
            if trans is not None:
                y_hat = trans.inverse_transform(y_hat)
        else:
            y_loss = batch.transform['y'].transform(y_loss)
        return self.loss_fn(y_hat, y_loss, batch.get('mask'))

    # ------------------------------------------------------------------
    # Training step: unpack CombinedLoader batch, dispatch to method
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            batch = batch[0]
        train_batch = batch["train"].to(self.device)
        val_batch = batch["val"].to(self.device)

        if self.rewire_every_k > 1 and (self.global_step % self.rewire_every_k != 0):
            return self._training_step_inner_only(train_batch)

        if self.bilevel_method == 'first_order':
            return self._training_step_first_order(train_batch, val_batch)
        elif self.bilevel_method == 'unrolled':
            return self._training_step_unrolled(train_batch, val_batch)
        elif self.bilevel_method == 'implicit':
            return self._training_step_implicit(train_batch, val_batch)
        else:
            raise ValueError(f"Unknown bilevel_method: {self.bilevel_method}")

    def _training_step_inner_only(self, train_batch):
        """Inner-loop only: update model params, skip policy (outer) update."""
        inner_opt = self.optimizers()[0]
        self.model.train()
        adj_s, adj_t = self._get_rewired_adj()
        self._inject_rewired_edges(train_batch, adj_s.detach())
        train_loss = self._compute_loss_from_batch(train_batch)
        inner_opt.zero_grad()
        self.manual_backward(train_loss)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        inner_opt.step()
        self.log('train_loss', train_loss, prog_bar=True)
        return train_loss

    # ------------------------------------------------------------------
    # Method 1: First-order approximation (DARTS-like)
    # ------------------------------------------------------------------

    def _training_step_first_order(self, train_batch, val_batch):
        inner_opt, outer_opt = self.optimizers()

        self.model.train()
        self.policy.train()
        # inner_batch_mode='reuse' (default): identical to the pre-ablation loop —
        # train_batch from CombinedLoader is held and reused across all T steps.
        # inner_batch_mode='fresh': each inner step draws the next batch from a
        # seeded cycling train loader (see on_train_epoch_start).
        fresh_fps = []
        for k in range(self.inner_steps):
            if self.inner_batch_mode == 'fresh':
                step_batch = self._next_fresh_batch()
                fresh_fps.append(self._batch_fingerprint(step_batch))
            else:
                step_batch = train_batch
            adj_s, adj_t = self._get_rewired_adj()
            self._inject_rewired_edges(step_batch, adj_s.detach())
            train_loss = self._compute_loss_from_batch(step_batch)
            inner_opt.zero_grad()
            self.manual_backward(train_loss)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            inner_opt.step()
        if self.inner_batch_mode == 'fresh' and self.inner_steps > 1:
            n_unique = len(set(fresh_fps))
            assert n_unique == len(fresh_fps), (
                f"inner_batch_mode=fresh but batches did not advance across "
                f"T={self.inner_steps} inner steps "
                f"(unique_fingerprints={n_unique}, fingerprints={fresh_fps})")

        self.model.eval()
        self.policy.train()
        adj_s, adj_t = self._get_rewired_adj()
        outer_batch = train_batch if self.outer_loss == 'train' else val_batch
        self._inject_rewired_edges(outer_batch, adj_s if not self.freeze_phi else adj_s.detach())
        val_loss = self._compute_loss_from_batch(outer_batch)

        if self.freeze_phi:
            policy_grad_norm = torch.tensor(0.0)
        else:
            outer_opt.zero_grad()
            self.manual_backward(val_loss)
            policy_grad_norm = sum(
                p.grad.norm() ** 2 for p in self.policy.parameters()
                if p.grad is not None) ** 0.5
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
            outer_opt.step()

        # Diagnostic: how much did the policy change the adjacency?
        with torch.no_grad():
            adj_rew, _ = self._get_rewired_adj()
            orig_nnz = int((self.adj_dense > 0).sum().item())
            rew_nnz = int((adj_rew > 0).sum().item())
            edges_added = rew_nnz - orig_nnz
            # Edge weight diff at original positions
            orig_mask = self.adj_dense > 0
            weight_diff = (adj_rew[orig_mask] - self.adj_dense[orig_mask]).abs().mean()
            # Scale diagnostics (for EdgeReweightPolicy)
            if orig_mask.any():
                scale_vals = adj_rew[orig_mask] / self.adj_dense[orig_mask].clamp(min=1e-8)
                scale_mean = scale_vals.mean()
                scale_std = scale_vals.std()
                scale_min = scale_vals.min()
                scale_max = scale_vals.max()
            else:
                scale_mean = scale_std = scale_min = scale_max = torch.tensor(0.0)
            # Relative edge diff: row-normalized difference (survives D^{-1})
            N = self.adj_dense.shape[0]
            orig_rsum = self.adj_dense.sum(dim=1, keepdim=True).clamp(min=1e-8)
            rew_rsum = adj_rew.sum(dim=1, keepdim=True).clamp(min=1e-8)
            orig_norm = self.adj_dense / orig_rsum
            rew_norm = adj_rew / rew_rsum
            relative_edge_diff = (rew_norm - orig_norm).abs().sum() / N

        # Corruption diagnostics: real vs fake edge scale
        real_edge_scale = fake_edge_scale = torch.tensor(0.0)
        if hasattr(self, 'adj_clean'):
            with torch.no_grad():
                clean_mask = self.adj_clean > 0
                fake_mask = (self.adj_dense > 0) & (~clean_mask)
                if clean_mask.any():
                    real_edge_scale = (adj_rew[clean_mask] / self.adj_dense[clean_mask].clamp(min=1e-8)).mean()
                if fake_mask.any():
                    fake_edge_scale = (adj_rew[fake_mask] / self.adj_dense[fake_mask].clamp(min=1e-8)).mean()

        self.log('train_loss', train_loss, prog_bar=True)
        self.log('val_loss_outer', val_loss, prog_bar=True)
        self.log('policy_grad_norm', policy_grad_norm)
        self.log('edges_added', float(edges_added))
        self.log('edge_weight_diff', weight_diff)
        self.log('scale_mean', scale_mean)
        self.log('scale_std', scale_std)
        self.log('scale_min', scale_min)
        self.log('scale_max', scale_max)
        self.log('relative_edge_diff', relative_edge_diff)
        if hasattr(self, 'adj_clean'):
            self.log('real_edge_scale', real_edge_scale)
            self.log('fake_edge_scale', fake_edge_scale)
        self.log('outer_inner_gap', val_loss - train_loss)
        # Edge addition diagnostics
        if hasattr(self.policy, '_last_edges_added') and self.policy._last_edges_added > 0:
            self.log('policy_edges_added', float(self.policy._last_edges_added))
            if self.policy._last_new_edge_weights is not None:
                self.log('new_edge_weight_mean', self.policy._last_new_edge_weights.mean())
                self.log('new_edge_weight_std', self.policy._last_new_edge_weights.std())
        return train_loss

    # ------------------------------------------------------------------
    # Method 2: Unrolled differentiation
    # ------------------------------------------------------------------

    def _training_step_unrolled(self, train_batch, val_batch):
        inner_opt, outer_opt = self.optimizers()

        self.model.train()
        self.policy.train()

        params = {n: p.clone() for n, p in self.model.named_parameters()}

        for _ in range(self.unroll_steps):
            adj_s, adj_t = self._get_rewired_adj()
            self._inject_rewired_edges(train_batch, adj_s)
            train_loss = self._compute_loss_functional(train_batch, params)
            grads = torch.autograd.grad(
                train_loss, list(params.values()), create_graph=True
            )
            params = {n: p - self.inner_lr * g
                      for (n, p), g in zip(params.items(), grads)}

        adj_s, adj_t = self._get_rewired_adj()
        self._inject_rewired_edges(val_batch, adj_s)
        val_loss = self._compute_loss_functional(val_batch, params)
        outer_opt.zero_grad()
        self.manual_backward(val_loss)
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        outer_opt.step()
        self.log('val_loss_outer', val_loss, prog_bar=True)

        adj_s_det, adj_t_det = self._get_rewired_adj()
        self._inject_rewired_edges(train_batch, adj_s_det.detach())
        train_loss_actual = self._compute_loss_from_batch(train_batch)
        inner_opt.zero_grad()
        self.manual_backward(train_loss_actual)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        inner_opt.step()

        self.log('train_loss', train_loss_actual, prog_bar=True)
        return train_loss_actual

    # ------------------------------------------------------------------
    # Method 3: Implicit differentiation (Neumann series)
    # ------------------------------------------------------------------

    def _training_step_implicit(self, train_batch, val_batch):
        inner_opt, outer_opt = self.optimizers()

        diff_params = [p for p in self.model.parameters() if p.requires_grad]

        self.model.train()
        self.policy.train()
        for _ in range(self.inner_steps):
            adj_s, adj_t = self._get_rewired_adj()
            self._inject_rewired_edges(train_batch, adj_s.detach())
            train_loss = self._compute_loss_from_batch(train_batch)
            inner_opt.zero_grad()
            self.manual_backward(train_loss)
            nn.utils.clip_grad_norm_(diff_params, self.grad_clip)
            inner_opt.step()

        self.model.eval()
        self.policy.train()

        adj_s, adj_t = self._get_rewired_adj()

        with torch.enable_grad():
            self._inject_rewired_edges(val_batch, adj_s)
            val_loss = self._compute_loss_from_batch(val_batch)
            d_val_d_theta = torch.autograd.grad(
                val_loss, diff_params,
                retain_graph=True, allow_unused=True,
            )
            d_val_d_theta = [g if g is not None else torch.zeros_like(p)
                             for g, p in zip(d_val_d_theta, diff_params)]

            self._inject_rewired_edges(train_batch, adj_s)
            train_loss_hv = self._compute_loss_from_batch(train_batch)
            grad_train = torch.autograd.grad(
                train_loss_hv, diff_params,
                create_graph=True, retain_graph=True,
            )

            # Neumann series: v ≈ H^{-1} g  where H = d²L_train/dθ², g = dL_val/dθ
            # Iteration: v_{k+1} = α·g + (I - α·H)·v_k,  v_0 = α·g
            v = [self.neumann_alpha * g.clone() for g in d_val_d_theta]
            for _ in range(self.neumann_steps):
                gv = sum((g * vi).sum() for g, vi in zip(grad_train, v))
                Hv = torch.autograd.grad(gv, diff_params, retain_graph=True)
                v = [self.neumann_alpha * dv + vi - self.neumann_alpha * hvi
                     for dv, vi, hvi in zip(d_val_d_theta, v, Hv)]

            gv_cross = sum((g * vi.detach()).sum() for g, vi in zip(grad_train, v))
            implicit_grad = torch.autograd.grad(
                gv_cross, self.policy.parameters(),
                retain_graph=True, allow_unused=True,
            )

            direct_grad = torch.autograd.grad(
                val_loss, self.policy.parameters(),
                retain_graph=False, allow_unused=True,
            )

        # dL_val/dφ = ∂L_val/∂φ  -  (∂²L_train/∂θ∂φ)^T · H^{-1} · ∂L_val/∂θ
        outer_opt.zero_grad()
        for p, dg, ig in zip(self.policy.parameters(), direct_grad, implicit_grad):
            dg_val = dg if dg is not None else torch.zeros_like(p)
            ig_val = ig if ig is not None else torch.zeros_like(p)
            p.grad = dg_val - ig_val

        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        outer_opt.step()

        self.log('train_loss', train_loss, prog_bar=True)
        self.log('val_loss_outer', val_loss, prog_bar=True)
        return train_loss

    # ------------------------------------------------------------------
    # Validation / Test — delegate to parent after injecting edges
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        self.policy.eval()
        adj_s, adj_t = self._get_rewired_adj()
        self._inject_rewired_edges(batch, adj_s.detach())
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.model.eval()
        self.policy.eval()
        adj_s, adj_t = self._get_rewired_adj()
        self._inject_rewired_edges(batch, adj_s.detach())
        if batch_idx == 0:
            with torch.no_grad():
                y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)
                y = batch.y
                assert y_hat.shape == y.shape, (
                    f"y_hat shape {y_hat.shape} != y shape {y.shape} — "
                    f"model may be collapsing the temporal horizon dim"
                )
                err = (y_hat - y).abs()
                horizon_mae = err.mean(dim=(0, 2, 3))
                pred_std = y_hat.std(dim=(0, 2, 3))
                print(f"[Phase0 diag] y_hat shape={y_hat.shape}, "
                      f"horizon MAE={horizon_mae.cpu().tolist()[:4]}..., "
                      f"pred std={pred_std.cpu().tolist()[:4]}...")
        return super().test_step(batch, batch_idx)

    @torch.no_grad()
    def get_rewired_adj(self):
        self.policy.eval()
        adj_s, adj_t, scores_s, scores_t = self.policy(self.adj_dense)
        return adj_s.cpu(), adj_t.cpu()
