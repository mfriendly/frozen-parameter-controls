"""Real-world spatiotemporal forecasting with graph rewiring.

Supports:
  - Static rewiring baselines: FoSR, SDRF, BORF, GTR (preprocessing)
  - Learned bilevel rewiring
  - No rewiring (vanilla baseline)

Usage:
  python experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=bilevel
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import numpy as np
import torch
import torch.serialization
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tsl import logger
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import scalers
from tsl.datasets import MetrLA, PemsBay
from tsl.engines import Predictor
from tsl.experiment import Experiment
from tsl.metrics import torch_metrics
from tsl.metrics.torch.metric_base import MaskedMetric
from tsl.metrics.torch.metrics import mape

# PyTorch 2.6+ defaults weights_only=True; whitelist tsl metric classes
from tsl.metrics.torch import metrics as _tsl_metrics
_safe_classes = [getattr(_tsl_metrics, c) for c in dir(_tsl_metrics)
                 if isinstance(getattr(_tsl_metrics, c, None), type)]
torch.serialization.add_safe_globals(_safe_classes)


class ThresholdMAPE(MaskedMetric):
    """MAPE that additionally masks out targets below a threshold (e.g. zero-flow)."""

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self, threshold=5.0, at=None, **kwargs):
        super().__init__(
            metric_fn=mape, mask_nans=False, mask_inf=True,
            metric_fn_kwargs={'reduction': 'none'}, at=at, **kwargs)
        self.threshold = threshold

    def update(self, y_hat, y, mask=None):
        thresh_mask = (y.abs() >= self.threshold)
        if mask is not None:
            mask = mask & thresh_mask
        else:
            mask = thresh_mask
        super().update(y_hat, y, mask)


torch.serialization.add_safe_globals([ThresholdMAPE])

import pytorch_lightning as pl
from lib.nn.models import get_model_class
from lib.nn.rewiring import apply_rewiring
from lib.utils import get_logger, finalize_experiment
from lib.utils.bottleneck import bottleneck_improvement


def corrupt_adj(adj_dense, corruption_ratio, seed=42):
    """Add random symmetric edges to adj_dense.

    corruption_ratio=0.3 means add 30% of original edge count as new edges.
    New edges receive the mean weight of existing edges.
    """
    g = torch.Generator().manual_seed(seed)
    N = adj_dense.shape[0]
    orig_mask = adj_dense > 0
    orig_nnz = int(orig_mask.sum().item())
    n_add = int(orig_nnz * corruption_ratio)
    non_edge_mask = (~orig_mask).clone()
    non_edge_mask.fill_diagonal_(False)
    idx = torch.arange(N)
    triu = non_edge_mask & (idx.unsqueeze(1) < idx.unsqueeze(0))
    candidates = torch.nonzero(triu, as_tuple=False)
    n_pairs = min(n_add // 2, candidates.shape[0])
    perm = torch.randperm(candidates.shape[0], generator=g)[:n_pairs]
    new_edges = candidates[perm]
    mean_w = adj_dense[orig_mask].mean().item()
    adj_corrupt = adj_dense.clone()
    adj_corrupt[new_edges[:, 0], new_edges[:, 1]] = mean_w
    adj_corrupt[new_edges[:, 1], new_edges[:, 0]] = mean_w
    return adj_corrupt


def corrupt_adj_swap(adj_dense, swap_ratio, seed=42):
    """Remove real edges and replace with same number of fake edges (symmetric).

    swap_ratio=0.2 means 20% of upper-triangle real edges are removed and
    replaced by random non-edges with mean original weight.
    """
    g = torch.Generator().manual_seed(seed)
    N = adj_dense.shape[0]
    idx = torch.arange(N)
    triu_mask = idx.unsqueeze(1) < idx.unsqueeze(0)
    orig_mask = adj_dense > 0
    orig_triu = orig_mask & triu_mask
    orig_edges = torch.nonzero(orig_triu, as_tuple=False)
    n_swap = int(orig_edges.shape[0] * swap_ratio)
    perm_rm = torch.randperm(orig_edges.shape[0], generator=g)[:n_swap]
    rm_edges = orig_edges[perm_rm]
    non_edge_mask = (~orig_mask).clone()
    non_edge_mask.fill_diagonal_(False)
    ne_triu = non_edge_mask & triu_mask
    candidates = torch.nonzero(ne_triu, as_tuple=False)
    perm_add = torch.randperm(candidates.shape[0], generator=g)[:n_swap]
    add_edges = candidates[perm_add]
    mean_w = adj_dense[orig_mask].mean().item()
    adj_out = adj_dense.clone()
    adj_out[rm_edges[:, 0], rm_edges[:, 1]] = 0
    adj_out[rm_edges[:, 1], rm_edges[:, 0]] = 0
    adj_out[add_edges[:, 0], add_edges[:, 1]] = mean_w
    adj_out[add_edges[:, 1], add_edges[:, 0]] = mean_w
    return adj_out


def _adj_to_dense_numpy(adj):
    import scipy.sparse as sp
    if sp.issparse(adj):
        return np.asarray(adj.todense(), dtype=np.float32)
    if isinstance(adj, torch.Tensor):
        return adj.detach().cpu().numpy().astype(np.float32)
    return np.asarray(adj, dtype=np.float32)


def _dense_to_adj_layout(dense, like):
    import scipy.sparse as sp
    if sp.issparse(like):
        return sp.csr_matrix(dense)
    if isinstance(like, torch.Tensor):
        return torch.tensor(dense, dtype=like.dtype)
    return dense


def adj_fingerprint(adj):
    """Stable hash of the nonzero pattern and weights (rounded)."""
    A = _adj_to_dense_numpy(adj)
    nz = np.nonzero(A)
    payload = np.concatenate([
        nz[0].astype(np.int64),
        nz[1].astype(np.int64),
        np.round(A[nz], 6).astype(np.float64).view(np.int64),
    ])
    return hash(payload.tobytes())


def adj_edge_stats(adj):
    A = _adj_to_dense_numpy(adj)
    mask = A > 0
    edge_count = int(mask.sum())
    mean_degree = float(mask.sum(axis=1).mean()) if A.shape[0] > 0 else 0.0
    return edge_count, mean_degree


def make_identity_adj(adj):
    """Identity adjacency in the same layout as `adj` (self-loops weight 1)."""
    A = _adj_to_dense_numpy(adj)
    N = A.shape[0]
    return _dense_to_adj_layout(np.eye(N, dtype=np.float32), adj)


def make_random_adj(adj, seed, n_sweeps=20):
    """Degree-sequence-preserving random graph via double-edge swaps.

    Topology is rewired with n_sweeps * E successful swaps (degree sequence
    exact). Edge weights are then randomly reassigned from the original
    multiset. Symmetric canonical -> undirected swaps + symmetrize; directed
    canonical -> directed swaps. No self-loops introduced.

    RNG: numpy Generator(seed + 104729) only — does not touch global
    numpy/torch streams used for weight init and batching.
    """
    A = _adj_to_dense_numpy(adj)
    N = A.shape[0]
    rng = np.random.default_rng(int(seed) + 104729)
    is_symmetric = bool(np.allclose(A, A.T, atol=1e-6))

    if is_symmetric:
        iu = np.triu_indices(N, k=1)
        mask_u = A[iu] > 0
        edges = np.stack([iu[0][mask_u], iu[1][mask_u]], axis=1).astype(np.int64)
        weights = A[iu][mask_u].copy()
        E = edges.shape[0]
        assert E > 1, f"need >=2 undirected edges to rewire, got {E}"
        edge_set = { (int(i), int(j)) for i, j in edges }
        target = n_sweeps * E
        done = 0
        trials = 0
        max_trials = target * 50
        while done < target and trials < max_trials:
            trials += 1
            a, b = rng.integers(0, E, size=2)
            if a == b:
                continue
            u, v = int(edges[a, 0]), int(edges[a, 1])
            x, y = int(edges[b, 0]), int(edges[b, 1])
            if len({u, v, x, y}) < 4:
                continue
            if rng.integers(0, 2) == 0:
                p1, p2 = (min(u, x), max(u, x)), (min(v, y), max(v, y))
            else:
                p1, p2 = (min(u, y), max(u, y)), (min(v, x), max(v, x))
            if p1[0] == p1[1] or p2[0] == p2[1]:
                continue
            if p1 in edge_set or p2 in edge_set:
                continue
            old1, old2 = (u, v), (x, y)
            edge_set.remove(old1)
            edge_set.remove(old2)
            edge_set.add(p1)
            edge_set.add(p2)
            edges[a, 0], edges[a, 1] = p1
            edges[b, 0], edges[b, 1] = p2
            done += 1
        assert done == target, (
            f"undirected swap mixing incomplete: {done}/{target} "
            f"(trials={trials}, E={E}, seed={seed})"
        )
        rng.shuffle(weights)
        out = np.zeros_like(A)
        out[edges[:, 0], edges[:, 1]] = weights
        out[edges[:, 1], edges[:, 0]] = weights
        deg_in = (A > 0).sum(1)
        deg_out = (out > 0).sum(1)
        assert np.array_equal(deg_in, deg_out), "degree sequence not preserved"
        return _dense_to_adj_layout(out, adj)

    # Directed: swap heads to preserve out- and in-degrees
    src, dst = np.nonzero(A)
    edges = np.stack([src, dst], axis=1).astype(np.int64)
    weights = A[src, dst].copy()
    E = edges.shape[0]
    assert E > 1, f"need >=2 directed edges to rewire, got {E}"
    edge_set = { (int(i), int(j)) for i, j in edges }
    target = n_sweeps * E
    done = 0
    trials = 0
    max_trials = target * 50
    while done < target and trials < max_trials:
        trials += 1
        a, b = rng.integers(0, E, size=2)
        if a == b:
            continue
        u, v = int(edges[a, 0]), int(edges[a, 1])
        x, y = int(edges[b, 0]), int(edges[b, 1])
        if u == x or v == y:
            continue
        p1, p2 = (u, y), (x, v)
        if p1[0] == p1[1] or p2[0] == p2[1]:
            continue
        if p1 in edge_set or p2 in edge_set:
            continue
        edge_set.remove((u, v))
        edge_set.remove((x, y))
        edge_set.add(p1)
        edge_set.add(p2)
        edges[a, 0], edges[a, 1] = p1
        edges[b, 0], edges[b, 1] = p2
        done += 1
    assert done == target, (
        f"directed swap mixing incomplete: {done}/{target} "
        f"(trials={trials}, E={E}, seed={seed})"
    )
    rng.shuffle(weights)
    out = np.zeros_like(A)
    out[edges[:, 0], edges[:, 1]] = weights
    assert np.array_equal((A > 0).sum(1), (out > 0).sum(1)), "out-degree not preserved"
    assert np.array_equal((A > 0).sum(0), (out > 0).sum(0)), "in-degree not preserved"
    return _dense_to_adj_layout(out, adj)


def _tensor_hash(t: torch.Tensor) -> str:
    import hashlib
    return hashlib.md5(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _tensor_hash_list(params: list) -> str:
    import hashlib
    h = hashlib.md5()
    for p in params:
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def get_agcrn_node_emb(model):
    """Return the AGCRN node-embedding Parameter, or None if absent."""
    agrn = getattr(model, 'agrn', None)
    if agrn is None:
        return None
    node_emb = getattr(agrn, 'node_emb', None)
    if node_emb is None:
        return None
    return node_emb.emb


def get_gwnet_embeddings(model) -> list:
    """Return [source_emb, target_emb] for GWNet, or [] if absent."""
    src = getattr(model, 'source_embeddings', None)
    tgt = getattr(model, 'target_embeddings', None)
    if src is None or tgt is None:
        return []
    s = getattr(src, 'emb', None)
    t = getattr(tgt, 'emb', None)
    if s is None or t is None:
        return []
    return [s, t]


def _get_adaptive_embedding_params(model) -> list:
    """Dispatch: return list of adaptive embedding Parameters for AGCRN or GWNet."""
    agcrn_emb = get_agcrn_node_emb(model)
    if agcrn_emb is not None:
        return [agcrn_emb]
    return get_gwnet_embeddings(model)


def apply_freeze_node_emb(predictor, freeze: bool):
    """Freeze/unfreeze adaptive node embeddings; when freeze, exclude from optimizer.

    Handles AGCRN (1 emb: node_emb) and GWNet (2 embs: source/target_embeddings).
    Parameter COUNT is unchanged; only requires_grad / optim membership change.
    Returns (emb_params, init_hash, n_trainable_frozen, n_trainable_unfrozen).
    """
    emb_params = _get_adaptive_embedding_params(predictor.model)
    if not emb_params:
        raise ValueError(
            "freeze_node_emb requires AGCRN (agrn.node_emb) or GWNet "
            "(source_embeddings / target_embeddings); "
            f"got {type(predictor.model).__name__}"
        )
    init_hash = _tensor_hash_list(emb_params)
    n_all = sum(p.numel() for p in predictor.model.parameters())
    n_emb = sum(p.numel() for p in emb_params)
    n_trainable_unfrozen = n_all
    n_trainable_frozen = n_all - n_emb
    emb_ids = {id(p) for p in emb_params}
    emb_desc = (
        f"node_emb{tuple(emb_params[0].shape)}" if len(emb_params) == 1 else
        f"source{tuple(emb_params[0].shape)}+target{tuple(emb_params[1].shape)}"
    )

    if freeze:
        for p in emb_params:
            p.requires_grad_(False)
            assert p.requires_grad is False
        _orig_configure = predictor.configure_optimizers

        def configure_optimizers_excl_frozen(self):
            cfg = dict()
            params = [p for p in self.parameters() if p.requires_grad]
            assert all(id(p) not in emb_ids for p in params), (
                "frozen emb still present in optimizer param list"
            )
            optimizer = self.optim_class(params, **self.optim_kwargs)
            cfg['optimizer'] = optimizer
            if self.scheduler_class is not None:
                metric = self.scheduler_kwargs.pop('monitor', None)
                scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
                cfg['lr_scheduler'] = scheduler
                if metric is not None:
                    cfg['monitor'] = metric
            return cfg

        import types
        predictor.configure_optimizers = types.MethodType(
            configure_optimizers_excl_frozen, predictor
        )
        predictor._freeze_node_emb_orig_configure = _orig_configure

    n_train_now = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
    expected_now = n_trainable_frozen if freeze else n_trainable_unfrozen
    assert n_train_now == expected_now, (n_train_now, expected_now, freeze)
    assert n_all == n_trainable_unfrozen

    logger.info(
        f"freeze_node_emb={freeze} n_params_total={n_all} "
        f"n_trainable_with_flag={n_trainable_frozen} "
        f"n_trainable_without_flag={n_trainable_unfrozen} "
        f"n_trainable_now={n_train_now} emb={emb_desc} "
        f"emb_init_hash={init_hash}"
    )
    return emb_params, init_hash, n_trainable_frozen, n_trainable_unfrozen


def get_dataset(dataset_cfg):
    name: str = dataset_cfg.name
    if name == 'la':
        dataset = MetrLA(impute_zeros=True)
    elif name == 'bay':
        dataset = PemsBay()
    elif name == 'pems04':
        from tsl.datasets import PeMS04
        dataset = PeMS04()
    elif name == 'pems07':
        from tsl.datasets import PeMS07
        dataset = PeMS07()
    elif name == 'pems08':
        from tsl.datasets import PeMS08
        dataset = PeMS08()
    elif name == 'airquality':
        from tsl.datasets import AirQuality
        dataset = AirQuality()
    elif name == 'engrad':
        from tsl.datasets import EngRad
        dataset = EngRad(**dataset_cfg.hparams)
        dataset.mask[:24] = True
    else:
        raise ValueError(f"Dataset {name} not available.")
    return dataset


def run(cfg: DictConfig):
    seed = cfg.get('seed') or 42
    pl.seed_everything(seed, workers=True)

    if cfg.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    ########################################
    # Get Dataset                          #
    ########################################

    dataset = get_dataset(cfg.dataset)
    adj = dataset.get_connectivity(**cfg.dataset.connectivity,
                                   layout=cfg.dataset.adj_layout)

    ########################################
    # Adjacency mode (canonical/identity/random)
    ########################################
    adjacency_mode = cfg.adjacency
    if adjacency_mode not in ('canonical', 'identity', 'random'):
        raise ValueError(
            f"adjacency must be one of canonical|identity|random, got {adjacency_mode!r}"
        )
    adj_canonical_fp = adj_fingerprint(adj)
    if adjacency_mode == 'identity':
        adj = make_identity_adj(adj)
    elif adjacency_mode == 'random':
        adj = make_random_adj(adj, seed=seed)
    edge_count, mean_degree = adj_edge_stats(adj)
    logger.info(
        f"Adjacency mode={adjacency_mode} edge_count={edge_count} "
        f"mean_degree={mean_degree:.4f}"
    )
    if adjacency_mode != 'canonical':
        adj_fp = adj_fingerprint(adj)
        assert adj_fp != adj_canonical_fp, (
            f"adjacency={adjacency_mode} produced an operator identical to canonical "
            f"(fingerprint={adj_fp})"
        )

    ########################################
    # Apply Rewiring (static methods)      #
    ########################################

    rewiring_cfg = cfg.get('rewiring', None)
    rewire_method = rewiring_cfg.method if rewiring_cfg else 'none'

    adj_orig = adj  # Keep original for analysis
    rewire_stats = {'method': 'none', 'edges_added': 0, 'rewire_time': 0.0}

    # External adjacency override (distillation / cross-backbone transfer)
    adj_load_path = cfg.get('adj_load', None)
    if adj_load_path is not None:
        import scipy.sparse as _sp_load
        if str(adj_load_path).endswith('.npy'):
            _adj_dense = np.load(adj_load_path).astype(np.float32)
        else:
            _adj_dense = torch.load(adj_load_path, map_location='cpu').float().numpy()
        adj = _sp_load.csr_matrix(_adj_dense)
        logger.info(f"Loaded external adj from {adj_load_path}, "
                    f"shape={adj.shape}, nnz={adj.nnz}")

    # Graph corruption (for robustness experiments)
    corruption_type = cfg.get('corruption_type', 'none')
    corruption_ratio = cfg.get('corruption_ratio', 0.0)
    adj_clean_dense = None
    if corruption_type != 'none' and corruption_ratio > 0:
        import scipy.sparse as _sp
        was_sparse = _sp.issparse(adj)
        if was_sparse:
            adj_d = torch.tensor(adj.todense(), dtype=torch.float32)
        else:
            adj_d = torch.tensor(adj, dtype=torch.float32)
        adj_clean_dense = adj_d.clone()
        orig_nnz = int((adj_clean_dense > 0).sum().item())
        if corruption_type == 'add':
            adj_d = corrupt_adj(adj_d, corruption_ratio, seed=42)
        elif corruption_type == 'swap':
            adj_d = corrupt_adj_swap(adj_d, corruption_ratio, seed=42)
        new_nnz = int((adj_d > 0).sum().item())
        logger.info(f"Graph corruption ({corruption_type}): ratio={corruption_ratio}, "
                    f"edges {orig_nnz} -> {new_nnz} (delta={new_nnz - orig_nnz})")
        adj = _sp.csr_matrix(adj_d.numpy()) if was_sparse else adj_d.numpy()

    # Global adjacency scaling (works for all code paths)
    global_adj_scale = cfg.get('adj_scale', None)
    if global_adj_scale is not None:
        import scipy.sparse as sp_sparse
        scale = float(global_adj_scale)
        if sp_sparse.issparse(adj):
            adj = adj.multiply(scale)
        elif isinstance(adj, np.ndarray):
            adj = adj * scale
        logger.info(f"Global adj_scale={scale} applied (nnz={adj.nnz if sp_sparse.issparse(adj) else int((adj != 0).sum())})")

    # Static pre-rewiring: apply FoSR/BORF/GTR before bilevel (or standalone)
    static_pre = cfg.get('static_pre_rewiring', None)
    if static_pre is not None:
        from omegaconf import OmegaConf
        pre_cfg = OmegaConf.create({'method': static_pre.method})
        pre_cfg.merge_with(static_pre)
        t_start = time.time()
        adj, _, pre_stats = apply_rewiring(adj, pre_cfg)
        pre_time = time.time() - t_start
        logger.info(f"Static pre-rewiring ({static_pre.method}): "
                    f"+{pre_stats.get('edges_added', 0)} edges in {pre_time:.2f}s")
        rewire_stats['pre_method'] = static_pre.method
        rewire_stats['pre_edges_added'] = pre_stats.get('edges_added', 0)

    if rewire_method not in ('none', 'bilevel', 'e2e_reweight', 'e2e_reweight_combined', 'drew', 'diag_vanilla_combined'):
        t_start = time.time()
        adj, edge_type, rewire_stats = apply_rewiring(adj, rewiring_cfg)
        rewire_stats['rewire_time'] = time.time() - t_start
        logger.info(f"Rewiring took {rewire_stats['rewire_time']:.2f}s")

    ########################################
    # Build Dataset & DataModule           #
    ########################################

    mask = dataset.mask

    # Covariates
    u = []
    if cfg.dataset.covariates.year:
        u.append(dataset.datetime_encoded('year').values)
    if cfg.dataset.covariates.day:
        u.append(dataset.datetime_encoded('day').values)
    if cfg.dataset.covariates.weekday:
        u.append(dataset.datetime_onehot('weekday').values)
    if cfg.dataset.covariates.mask:
        u.append(mask.astype(np.float32))
    if 'u' in dataset.covariates:
        u.append(dataset.get_frame('u', return_pattern=False))

    if len(u):
        ndim = max(u_.ndim for u_ in u)
        u = np.concatenate([np.repeat(u_[:, None], dataset.n_nodes, 1)
                            if u_.ndim < ndim else u_
                            for u_ in u], axis=-1)
        covariates = dict(u=u)
    else:
        covariates = None

    temporal_dilation = cfg.get('temporal_dilation', 1)
    effective_window = cfg.window * temporal_dilation

    torch_dataset = SpatioTemporalDataset(target=dataset.dataframe(),
                                          mask=mask,
                                          covariates=covariates,
                                          connectivity=adj,
                                          horizon=cfg.horizon,
                                          window=effective_window,
                                          stride=cfg.stride)

    if temporal_dilation > 1:
        _dil = temporal_dilation
        _ew = effective_window

        def _dilate_transform(sample):
            for group_name in ('input', 'target', 'auxiliary'):
                group = getattr(sample, group_name, None)
                if group is None:
                    continue
                for key in list(group.keys()):
                    val = group[key]
                    if not isinstance(val, torch.Tensor) or val.dim() == 0:
                        continue
                    if val.size(0) == _ew:
                        group[key] = val[::_dil]
            return sample

        torch_dataset.transform = _dilate_transform
        logger.info(f"Temporal dilation={temporal_dilation}: "
                     f"window {effective_window} -> subsampled to {cfg.window} steps "
                     f"(effective coverage = {effective_window} time steps)")

    # Scale input features
    scaler_cfg = cfg.get('scaler')
    if scaler_cfg is not None:
        scale_axis = (0,) if scaler_cfg.axis == 'node' else (0, 1)
        scaler_cls = getattr(scalers, f'{scaler_cfg.method}Scaler')
        transform = dict(target=scaler_cls(axis=scale_axis))
    else:
        transform = None

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
        mask_scaling=True,
        batch_size=cfg.batch_size,
        workers=cfg.workers
    )
    dm.setup()

    ########################################
    # Create Model                         #
    ########################################

    model_cls = get_model_class(cfg.model.name)

    d_exog = torch_dataset.input_map.u.shape[-1] if 'u' in torch_dataset else 0
    model_kwargs = dict(n_nodes=dm.n_nodes,
                        input_size=dm.n_channels,
                        mask_size=dm.n_channels,
                        exog_size=d_exog,
                        output_size=cfg.get("out_channels", dm.n_channels),
                        horizon=dm.horizon,
                        input_encoding=cfg.input_encoding)

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    ########################################
    # Create Predictor                     #
    ########################################

    loss_fn = getattr(torch_metrics, f"Masked{cfg.loss_fn.upper()}")()

    log_metrics = {
        'mae': torch_metrics.MaskedMAE(),
        'mse': torch_metrics.MaskedMSE(),
        'mape': ThresholdMAPE(threshold=5.0),
    }
    for h in [3, 6, 9, 12]:
        idx = h - 1
        if idx < dm.horizon:
            log_metrics[f'mae_h{h}'] = torch_metrics.MaskedMAE(at=idx)

    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    if rewire_method in ('bilevel', 'e2e_reweight', 'e2e_reweight_combined'):
        # Bilevel or E2E rewiring: build policy + predictor
        from lib.nn.rewiring.bilevel import (BilevelPredictor, E2EReweightPredictor,
                                              E2EReweightCombinedPredictor)

        # Convert adj to dense for policy — preserve original edge weights
        import scipy.sparse as sp_sparse
        if sp_sparse.issparse(adj):
            adj_dense = torch.tensor(adj.todense(), dtype=torch.float32)
            num_nodes = adj.shape[0]
        elif isinstance(adj, np.ndarray):
            adj_dense = torch.tensor(adj, dtype=torch.float32)
            num_nodes = adj.shape[0]
        else:
            from lib.nn.rewiring import adj_to_edge_index
            ei_np, num_nodes = adj_to_edge_index(adj)
            adj_dense = torch.zeros(num_nodes, num_nodes)
            for idx in range(ei_np.shape[1]):
                adj_dense[ei_np[0, idx], ei_np[1, idx]] = 1.0

        adj_load_path = rewiring_cfg.get('adj_load', None)
        if adj_load_path is not None:
            if str(adj_load_path).endswith('.npy'):
                adj_dense = torch.from_numpy(np.load(adj_load_path)).float()
            else:
                adj_dense = torch.load(adj_load_path, map_location='cpu').float()
            logger.info(f"Loaded adj from {adj_load_path}, nnz={int((adj_dense > 0).sum())}")
        adj_scale = rewiring_cfg.get('adj_scale', None)
        if adj_scale is not None:
            adj_dense = adj_dense * float(adj_scale)
            logger.info(f"Scaled adj by {adj_scale}, nnz={int((adj_dense != 0).sum())}")

        policy_type = rewiring_cfg.get('policy_type', 'joint_rewire')
        if rewiring_cfg.get('policy_identity', False):
            from lib.nn.rewiring.policy import IdentityPolicy
            policy = IdentityPolicy(seq_len=cfg.window)
            logger.info("Using IDENTITY policy (diagnostic mode)")
        elif policy_type == 'edge_reweight':
            from lib.nn.rewiring.policy import EdgeReweightPolicy
            policy = EdgeReweightPolicy(
                num_nodes=num_nodes,
                adj_dense=adj_dense,
                seq_len=cfg.window,
                hidden_dim=rewiring_cfg.get('policy_hidden', 64),
                num_new_edges=rewiring_cfg.get('policy_num_new_edges', 0),
                temperature=rewiring_cfg.get('policy_temperature', 1.0),
            )
            logger.info(f"Using EdgeReweightPolicy (hidden={rewiring_cfg.get('policy_hidden', 64)}, "
                        f"new_edges={rewiring_cfg.get('policy_num_new_edges', 0)}, "
                        f"temp={rewiring_cfg.get('policy_temperature', 1.0)})")
        elif policy_type == 'relative_reweight':
            from lib.nn.rewiring.policy import RelativeReweightPolicy
            policy = RelativeReweightPolicy(
                num_nodes=num_nodes,
                adj_dense=adj_dense,
                seq_len=cfg.window,
                hidden_dim=rewiring_cfg.get('policy_hidden', 64),
                temperature=rewiring_cfg.get('policy_temperature', 1.0),
                num_new_edges=rewiring_cfg.get('policy_num_new_edges', 0),
                disable_reweight=rewiring_cfg.get('disable_reweight', False),
            )
            logger.info(f"Using RelativeReweightPolicy (hidden={rewiring_cfg.get('policy_hidden', 64)}, "
                        f"temp={rewiring_cfg.get('policy_temperature', 1.0)}, "
                        f"num_new_edges={rewiring_cfg.get('policy_num_new_edges', 0)})")
        else:
            from lib.nn.rewiring.policy import JointRewirePolicy
            policy = JointRewirePolicy(
                num_nodes=num_nodes,
                seq_len=cfg.window,
                hidden_dim=rewiring_cfg.policy_hidden,
                num_candidates=rewiring_cfg.num_candidates,
                topk_spatial=rewiring_cfg.topk_spatial,
                topk_temporal=rewiring_cfg.topk_temporal,
                tau=rewiring_cfg.tau,
            )

        if rewire_method in ('e2e_reweight', 'e2e_reweight_combined'):
            cls = (E2EReweightCombinedPredictor if rewire_method == 'e2e_reweight_combined'
                   else E2EReweightPredictor)
            predictor = cls(
                model_class=model_cls,
                model_kwargs=model_kwargs,
                policy=policy,
                adj_dense=adj_dense,
                grad_clip=cfg.grad_clip_val,
                optim_class=getattr(torch.optim, cfg.optimizer.name),
                optim_kwargs=dict(cfg.optimizer.hparams),
                loss_fn=loss_fn,
                metrics=log_metrics,
                scheduler_class=scheduler_class,
                scheduler_kwargs=scheduler_kwargs,
                scale_target=False if scaler_cfg is None else scaler_cfg.scale_target,
            )
            logger.info(f"{rewire_method}: policy params added to main optimizer")
        else:
            predictor = BilevelPredictor(
                model_class=model_cls,
                model_kwargs=model_kwargs,
                policy=policy,
                adj_dense=adj_dense,
                bilevel_method=rewiring_cfg.get('bilevel_method', 'first_order'),
                inner_lr=cfg.optimizer.hparams.lr,
                outer_lr=rewiring_cfg.outer_lr,
                inner_steps=rewiring_cfg.get('inner_steps', 1),
                unroll_steps=rewiring_cfg.get('unroll_steps', 5),
                neumann_steps=rewiring_cfg.get('neumann_steps', 5),
                neumann_alpha=rewiring_cfg.get('neumann_alpha', 0.1),
                grad_clip=cfg.grad_clip_val,
                outer_warmup_epochs=rewiring_cfg.get('outer_warmup_epochs', 0),
                outer_cosine=rewiring_cfg.get('outer_cosine', False),
                max_epochs=cfg.epochs,
                rewire_every_k=rewiring_cfg.get('rewire_every_k', 1),
                freeze_phi=rewiring_cfg.get('freeze_phi', False),
                outer_loss=rewiring_cfg.get('outer_loss', 'val'),
                inner_batch_mode=rewiring_cfg.inner_batch_mode,
                optim_class=getattr(torch.optim, cfg.optimizer.name),
                optim_kwargs=dict(cfg.optimizer.hparams),
                loss_fn=loss_fn,
                metrics=log_metrics,
                scheduler_class=scheduler_class,
                scheduler_kwargs=scheduler_kwargs,
                scale_target=False if scaler_cfg is None else scaler_cfg.scale_target,
            )
            if adj_clean_dense is not None:
                predictor.register_buffer('adj_clean', adj_clean_dense)
            predictor.configure_inner_batch_source(
                dataset=dm.trainset,
                batch_size=cfg.batch_size,
                seed=cfg.seed,
            )
            freeze_phi = rewiring_cfg.get('freeze_phi', False)
            outer_loss = rewiring_cfg.get('outer_loss', 'val')
            inner_batch_mode = rewiring_cfg.inner_batch_mode
            logger.info(f"Bilevel method: {rewiring_cfg.bilevel_method}"
                        f"{' [FROZEN PHI]' if freeze_phi else ''}"
                        f"{' [OUTER=TRAIN]' if outer_loss == 'train' else ''}"
                        f" [inner_batch_mode={inner_batch_mode}]")
        # Disable GNN caches: learned rewiring changes edge weights every step.
        for module in predictor.model.modules():
            if hasattr(module, 'cached'):
                module.cached = False
                if hasattr(module, '_cached_edge_index'):
                    module._cached_edge_index = None
                if hasattr(module, '_cached_adj_t'):
                    module._cached_adj_t = None
            if hasattr(module, '_support'):
                module._support = None
    elif rewire_method == 'diag_vanilla_combined':
        # Diagnostic: vanilla Predictor + CombinedLoader → isolates data volume effect
        from lib.nn.rewiring.bilevel import VanillaCombinedPredictor
        predictor = VanillaCombinedPredictor(
            model_class=model_cls,
            model_kwargs=model_kwargs,
            optim_class=getattr(torch.optim, cfg.optimizer.name),
            optim_kwargs=dict(cfg.optimizer.hparams),
            loss_fn=loss_fn,
            metrics=log_metrics,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            scale_target=False if scaler_cfg is None else scaler_cfg.scale_target,
        )
        logger.info("DIAG: VanillaCombinedPredictor (auto-opt + CombinedLoader train key)")
    else:
        # Standard predictor (vanilla or static rewiring)
        predictor = Predictor(
            model_class=model_cls,
            model_kwargs=model_kwargs,
            optim_class=getattr(torch.optim, cfg.optimizer.name),
            optim_kwargs=dict(cfg.optimizer.hparams),
            loss_fn=loss_fn,
            metrics=log_metrics,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            scale_target=False if scaler_cfg is None else scaler_cfg.scale_target,
        )

    ########################################
    # AGCRN node-embedding freeze control  #
    ########################################
    freeze_node_emb = bool(cfg.freeze_node_emb)
    emb_params_list = None
    emb_init_hash = None
    n_trainable_params = None
    _has_adaptive_emb = bool(_get_adaptive_embedding_params(predictor.model))
    if freeze_node_emb and not _has_adaptive_emb:
        raise ValueError(
            f"freeze_node_emb=true but model {type(predictor.model).__name__} "
            "has no AGCRN node_emb or GWNet source/target_embeddings"
        )
    if _has_adaptive_emb:
        # AGCRN or GWNet: log side-by-side counts + track emb hash.
        # freeze=False leaves requires_grad and configure_optimizers untouched.
        emb_params_list, emb_init_hash, n_frz, n_unfrz = apply_freeze_node_emb(
            predictor, freeze=freeze_node_emb
        )
        n_trainable_params = n_frz if freeze_node_emb else n_unfrz
        logger.info(
            f"trainable_params side-by-side: "
            f"with_freeze={n_frz} without_freeze={n_unfrz}"
        )

    # Unconditional trainable-param count of the actual built model (includes exog covariates),
    # so no-adaptive runs (learned_adjacency=False, where the freeze block above is skipped)
    # still record a count. Same definition as the freeze block: requires_grad params.
    n_trainable_total = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
    logger.info(f"n_trainable_total={n_trainable_total}")

    ########################################
    # Logging                              #
    ########################################

    exp_logger = get_logger(exp, cfg, **rewire_stats)

    ########################################
    # Training                             #
    ########################################

    # Vanilla and bilevel both run the validation loop and monitor the same metric
    monitor_metric = f'val_{cfg.loss_fn.lower()}'
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=cfg.patience,
        mode='min',
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor=monitor_metric,
        mode='min',
    )

    callbacks = [early_stop_callback, checkpoint_callback]
    if emb_params_list is not None:
        from pytorch_lightning import Callback as _PLCallback

        class _NodeEmbGuard(_PLCallback):
            def __init__(self, emb_list, init_hash, freeze):
                self.emb_list = emb_list
                self.init_hash = init_hash
                self.freeze = freeze

            def on_fit_start(self, trainer, pl_module):
                opt = trainer.optimizers[0]
                ids = {id(p) for g in opt.param_groups for p in g['params']}
                emb_ids = {id(e) for e in self.emb_list}
                if self.freeze:
                    assert not emb_ids.intersection(ids), (
                        "frozen emb is in the optimizer param_groups"
                    )
                else:
                    assert emb_ids.issubset(ids), (
                        "trainable emb missing from optimizer param_groups"
                    )

            def on_train_end(self, trainer, pl_module):
                end_hash = _tensor_hash_list(self.emb_list)
                logger.info(
                    f"node_emb_hash init={self.init_hash} end={end_hash} "
                    f"freeze_node_emb={self.freeze}"
                )
                if self.freeze:
                    assert end_hash == self.init_hash, (
                        f"frozen emb changed during training: "
                        f"init={self.init_hash} end={end_hash}"
                    )
                else:
                    assert end_hash != self.init_hash, (
                        f"trainable emb did not change during training: "
                        f"hash={end_hash}"
                    )

        callbacks.append(_NodeEmbGuard(emb_params_list, emb_init_hash, freeze_node_emb))

    # Manual-optimization predictors handle their own grad clipping.
    manual_opt = rewire_method in ('bilevel', 'e2e_reweight_combined')
    clip_val = None if manual_opt else cfg.grad_clip_val
    trainer = Trainer(max_epochs=cfg.epochs,
                      limit_train_batches=cfg.get("train_batches", None),
                      default_root_dir=cfg.run.dir,
                      logger=exp_logger,
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=1,
                      gradient_clip_val=clip_val,
                      deterministic=cfg.deterministic,
                      callbacks=callbacks)

    load_model_path = cfg.get('load_model_path')
    if load_model_path is not None:
        predictor.load_model(load_model_path)
    else:
        if rewire_method in ('bilevel', 'e2e_reweight_combined', 'diag_vanilla_combined'):
            from lib.nn.rewiring.bilevel import make_bilevel_loader
            combined_dl = make_bilevel_loader(dm.train_dataloader(),
                                              dm.val_dataloader())
            trainer.fit(predictor,
                        train_dataloaders=combined_dl,
                        val_dataloaders=dm.val_dataloader())
        else:
            trainer.fit(predictor,
                        train_dataloaders=dm.train_dataloader(),
                        val_dataloaders=dm.val_dataloader())
        predictor.load_model(checkpoint_callback.best_model_path)
        if emb_params_list is not None and freeze_node_emb:
            # Best-ckpt reload must still match the frozen init embedding.
            end_hash = _tensor_hash_list(_get_adaptive_embedding_params(predictor.model))
            assert end_hash == emb_init_hash, (
                f"frozen emb changed after ckpt load: "
                f"init={emb_init_hash} end={end_hash}"
            )

    predictor.freeze()

    ########################################
    # Testing                              #
    ########################################

    trainer.test(predictor, dataloaders=dm.test_dataloader())

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"Peak GPU memory: {peak_gb:.1f} GB")

    # Auto-append this run to results.csv immediately after test
    from collect_results import append_run
    append_run(Path(cfg.run.dir))

    ########################################
    # Bottleneck Analysis                  #
    ########################################

    if rewire_method not in ('none',) and rewire_method != 'drew':
        gnn_layers = cfg.model.hparams.get('gnn_layers', 1)

        if rewire_method in ('bilevel', 'e2e_reweight', 'e2e_reweight_combined'):
            adj_rewired_s, adj_rewired_t = predictor.get_rewired_adj()
            adj_save_path = rewiring_cfg.get('adj_save', None)
            if adj_save_path is not None:
                torch.save(adj_rewired_s.detach().cpu(), adj_save_path)
                logger.info(f"Saved learned adj to {adj_save_path}")
            # Adj diff stats (diagnostic #1) — count non-zero edges
            orig_edges = int((adj_dense > 0).sum().item())
            rewired_edges = int((adj_rewired_s > 0).sum().item())
            diff = (adj_rewired_s > 0).float() - (adj_dense > 0).float()
            added = int((diff > 0).sum().item())
            removed = int((diff < 0).sum().item())
            logger.info(f"[Adj stats] orig_edges={orig_edges} rewired_edges={rewired_edges} "
                         f"added={added} removed={removed}")
            improvement = bottleneck_improvement(
                adj_dense.numpy(), adj_rewired_s.numpy(), gnn_layers
            )
        else:
            from lib.nn.rewiring import adj_to_edge_index, edge_index_to_sparse
            ei_orig, n = adj_to_edge_index(adj_orig)
            adj_orig_dense = np.zeros((n, n))
            for idx in range(ei_orig.shape[1]):
                adj_orig_dense[ei_orig[0, idx], ei_orig[1, idx]] = 1.0

            ei_new, n = adj_to_edge_index(adj)
            adj_new_dense = np.zeros((n, n))
            for idx in range(ei_new.shape[1]):
                adj_new_dense[ei_new[0, idx], ei_new[1, idx]] = 1.0

            improvement = bottleneck_improvement(
                adj_orig_dense, adj_new_dense, gnn_layers
            )

        logger.info(f"Spectral gap: {improvement['spectral_gap_orig']:.4f} "
                     f"-> {improvement['spectral_gap_rewired']:.4f} "
                     f"(+{improvement['spectral_gap_improvement']:.4f})")
        logger.info(f"Total resistance: {improvement['total_resistance_orig']:.1f} "
                     f"-> {improvement['total_resistance_rewired']:.1f} "
                     f"(-{improvement['total_resistance_reduction']:.1f})")

    finalize_experiment(exp_logger)

    result = checkpoint_callback.best_model_score
    if result is not None:
        result = result.item()
    return result


if __name__ == '__main__':
    exp = Experiment(run_fn=run, config_path='../config/',
                     config_name='realworld_rewired')
    res = exp.run()
    logger.info(res)
