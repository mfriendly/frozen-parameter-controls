"""Learned rewiring policies for bilevel optimization.

SpatialRewirePolicy:  learns which spatial edges to add via node embeddings.
TemporalRewirePolicy: learns temporal skip connections between time steps.
JointRewirePolicy:    combines spatial and temporal rewiring.
EdgeReweightPolicy:   multiplicative reweighting of existing edges + optional new edges.

All policies use Gumbel-softmax for differentiable discrete edge selection
during training, and hard thresholding during inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialRewirePolicy(nn.Module):
    """Learns to add spatial edges via node embedding similarity.

    For each forward pass:
      1. Sample candidate non-edges from the complement graph
      2. Score candidates using learned node embeddings
      3. Select top-k via Gumbel-softmax (training) or hard threshold (eval)
      4. Return rewired adjacency

    Args:
        num_nodes: number of nodes in the spatial graph.
        hidden_dim: embedding dimension.
        num_candidates: number of candidate edges to sample per forward pass.
        topk: number of edges to actually add.
        tau: Gumbel-softmax temperature.
    """

    def __init__(self, num_nodes, hidden_dim, num_candidates, topk, tau=0.5):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_candidates = num_candidates
        self.topk = topk
        self.tau = tau

        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _get_candidates(self, adj):
        """Sample candidate non-edges from complement graph."""
        non_edges = (adj == 0).float()
        non_edges.fill_diagonal_(0)
        flat = non_edges.view(-1)
        num_cand = min(self.num_candidates, int(flat.sum().item()))
        if num_cand == 0:
            return torch.zeros(0, dtype=torch.long, device=adj.device)
        indices = flat.nonzero(as_tuple=False).squeeze(-1)
        perm = torch.randperm(indices.size(0), device=adj.device)[:num_cand]
        return indices[perm]

    def forward(self, adj):
        """
        Args:
            adj: (N, N) dense adjacency tensor.

        Returns:
            adj_rewired: (N, N) rewired adjacency.
            scores: (num_candidates,) raw edge scores.
        """
        cand_flat = self._get_candidates(adj)
        if cand_flat.numel() == 0:
            return adj, torch.zeros(0, device=adj.device)

        rows = cand_flat // self.num_nodes
        cols = cand_flat % self.num_nodes

        emb_r = self.node_emb(rows)
        emb_c = self.node_emb(cols)
        scores = self.edge_scorer(
            torch.cat([emb_r, emb_c], dim=-1)
        ).squeeze(-1)

        if self.training:
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(scores) + 1e-20) + 1e-20
            )
            soft = torch.sigmoid((scores + gumbel_noise) / self.tau)
        else:
            soft = (scores > 0).float()

        k = min(self.topk, soft.size(0))
        topk_vals, topk_idx = soft.topk(k)

        adj_rewired = adj.clone()
        sel_rows = rows[topk_idx]
        sel_cols = cols[topk_idx]
        adj_rewired[sel_rows, sel_cols] = topk_vals
        adj_rewired[sel_cols, sel_rows] = topk_vals

        return adj_rewired, scores


class TemporalRewirePolicy(nn.Module):
    """Learns temporal skip connections between time steps.

    Builds on the causal temporal adjacency (t+1 -> t) and adds
    learned long-range skip connections (t -> t-k for k > 1).

    Args:
        seq_len: input sequence length.
        hidden_dim: embedding dimension.
        topk: number of skip connections to add.
        tau: Gumbel-softmax temperature.
    """

    def __init__(self, seq_len, hidden_dim, topk, tau=0.5):
        super().__init__()
        self.seq_len = seq_len
        self.topk = topk
        self.tau = tau

        self.time_emb = nn.Embedding(seq_len, hidden_dim)
        self.skip_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self):
        """
        Returns:
            adj_t: (T, T) temporal adjacency with skip connections.
            scores: raw skip connection scores.
        """
        device = self.time_emb.weight.device
        idx = torch.arange(self.seq_len, device=device)
        emb = self.time_emb(idx)

        # All causal pairs (i < j) as candidates for skip connections
        ii = idx.unsqueeze(1).expand(-1, self.seq_len).reshape(-1)
        jj = idx.unsqueeze(0).expand(self.seq_len, -1).reshape(-1)
        mask = (ii < jj)
        ii, jj = ii[mask], jj[mask]

        dt = (jj - ii).float().unsqueeze(-1) / self.seq_len
        pair_feat = torch.cat([emb[ii], emb[jj], dt], dim=-1)
        scores = self.skip_scorer(pair_feat).squeeze(-1)

        if self.training:
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(scores) + 1e-20) + 1e-20
            )
            soft = torch.sigmoid((scores + gumbel_noise) / self.tau)
        else:
            soft = (scores > 0).float()

        # Base causal adjacency: t+1 depends on t
        base = torch.zeros(self.seq_len, self.seq_len, device=device)
        for t in range(self.seq_len - 1):
            base[t + 1, t] = 1.0

        # Add top-k skip connections
        k = min(self.topk, soft.size(0))
        topk_vals, topk_idx = soft.topk(k)
        base[jj[topk_idx], ii[topk_idx]] = topk_vals

        return base, scores


class IdentityPolicy(nn.Module):
    """Diagnostic policy: returns original adjacency unchanged.

    Used to verify BilevelPredictor training loop matches vanilla Predictor
    by eliminating rewiring as a variable.
    """

    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        # Dummy param so outer optimizer doesn't crash (gets zero grad)
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, adj_spatial):
        adj_s = adj_spatial.clone()
        adj_t = torch.zeros(self.seq_len, self.seq_len, device=adj_spatial.device)
        for t in range(self.seq_len - 1):
            adj_t[t + 1, t] = 1.0
        scores_s = torch.zeros(0, device=adj_spatial.device)
        scores_t = torch.zeros(0, device=adj_spatial.device)
        return adj_s, adj_t, scores_s, scores_t


class JointRewirePolicy(nn.Module):
    """Joint spatial + temporal rewiring policy.

    Args:
        num_nodes: number of spatial nodes.
        seq_len: input sequence length.
        hidden_dim: policy embedding dimension.
        num_candidates: spatial candidate edges to sample.
        topk_spatial: spatial edges to add.
        topk_temporal: temporal skip connections to add.
        tau: Gumbel-softmax temperature.
    """

    def __init__(self, num_nodes, seq_len, hidden_dim, num_candidates,
                 topk_spatial, topk_temporal, tau=0.5):
        super().__init__()
        self.spatial = SpatialRewirePolicy(
            num_nodes, hidden_dim, num_candidates, topk_spatial, tau
        )
        self.temporal = TemporalRewirePolicy(
            seq_len, hidden_dim, topk_temporal, tau
        )

    def forward(self, adj_spatial):
        """
        Args:
            adj_spatial: (N, N) spatial adjacency.

        Returns:
            adj_s: rewired spatial adjacency.
            adj_t: temporal adjacency with skip connections.
            scores_s: spatial edge scores.
            scores_t: temporal skip scores.
        """
        adj_s, scores_s = self.spatial(adj_spatial)
        adj_t, scores_t = self.temporal()
        return adj_s, adj_t, scores_s, scores_t


class EdgeReweightPolicy(nn.Module):
    """Multiplicative reweighting of existing edges + optional new edge addition.

    Architecture:
      1. Node embedding: MLP(adj_row) -> h_i for each node
      2. Edge score: s_ij = MLP([h_i; h_j; orig_w_ij]) for existing edges
      3. New weight: w_ij = orig_w_ij * 2 * sigmoid(s_ij / temperature)
         - At init: score ~ 0 -> 2*sigmoid(0) = 1.0 -> identity
         - Range: (0, 2*orig_w) -> can downweight or upweight
      4. Optional new edges: score non-edges via separate head, top-k addition

    Args:
        num_nodes: number of spatial nodes.
        adj_dense: (N, N) original dense adjacency tensor.
        seq_len: input sequence length (for temporal adj placeholder).
        hidden_dim: policy embedding dimension.
        num_new_edges: number of new edges to add (0 = reweight only).
        temperature: softmax temperature for score scaling.
    """

    def __init__(self, num_nodes, adj_dense, seq_len, hidden_dim=64,
                 num_new_edges=0, temperature=1.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.num_new_edges = num_new_edges
        self.temperature = temperature

        # Cache original edge positions and weights
        self.register_buffer('orig_adj', adj_dense.clone())
        row, col = torch.nonzero(adj_dense, as_tuple=True)
        self.register_buffer('edge_row', row)
        self.register_buffer('edge_col', col)
        self.register_buffer('orig_weights', adj_dense[row, col].clone())

        # Cache non-edge candidates (upper-triangle only, symmetrized in forward)
        non_edge_mask = (adj_dense == 0)
        non_edge_mask.fill_diagonal_(False)
        idx = torch.arange(num_nodes)
        triu_mask = non_edge_mask & (idx.unsqueeze(1) < idx.unsqueeze(0))
        ne_row, ne_col = torch.nonzero(triu_mask, as_tuple=True)
        self.register_buffer('ne_row', ne_row)
        self.register_buffer('ne_col', ne_col)

        # Node encoder: adj_row -> embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Reweight head: [h_i; h_j; orig_w] -> score
        self.reweight_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Zero-init final layer -> sigmoid(0)=0.5 -> scale=1.0 -> identity
        nn.init.zeros_(self.reweight_head[-1].weight)
        nn.init.zeros_(self.reweight_head[-1].bias)

        # Edge addition head (only if num_new_edges > 0)
        if num_new_edges > 0:
            self.add_head = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, adj_dense):
        """
        Args:
            adj_dense: (N, N) dense adjacency (stored buffer, used for node features).

        Returns:
            adj_s: (N, N) reweighted spatial adjacency.
            adj_t: (T, T) temporal adjacency (causal baseline, not learned here).
            scores_s: (E,) raw reweight scores for existing edges.
            scores_t: empty tensor (temporal not learned in this policy).
        """
        N = self.num_nodes
        device = adj_dense.device
        h = self.node_encoder(adj_dense)  # [N, hidden_dim]

        # --- Reweight existing edges ---
        h_i = h[self.edge_row]
        h_j = h[self.edge_col]
        edge_feat = torch.cat([h_i, h_j, self.orig_weights.unsqueeze(-1)], dim=-1)
        scores = self.reweight_head(edge_feat).squeeze(-1)
        # Multiplicative scale: 2*sigmoid(score/T). At init: score~0 -> scale=1.0
        scale = 2.0 * torch.sigmoid(scores / self.temperature)
        new_weights = self.orig_weights * scale

        adj_out = torch.zeros(N, N, device=device, dtype=adj_dense.dtype)
        adj_out[self.edge_row, self.edge_col] = new_weights

        # --- Optionally add new edges ---
        if self.num_new_edges > 0 and self.ne_row.shape[0] > 0:
            h_ne_i = h[self.ne_row]
            h_ne_j = h[self.ne_col]
            ne_scores = self.add_head(
                torch.cat([h_ne_i, h_ne_j], dim=-1)
            ).squeeze(-1)
            k = min(self.num_new_edges, ne_scores.shape[0])
            topk_vals, topk_idx = torch.topk(ne_scores, k)
            ne_weights = torch.sigmoid(topk_vals / self.temperature)
            ne_weights = ne_weights * self.orig_weights.mean()
            sel_row = self.ne_row[topk_idx]
            sel_col = self.ne_col[topk_idx]
            adj_out[sel_row, sel_col] = ne_weights
            adj_out[sel_col, sel_row] = ne_weights

        # Temporal: simple causal baseline (not learned)
        adj_t = torch.zeros(self.seq_len, self.seq_len, device=device)
        for t in range(self.seq_len - 1):
            adj_t[t + 1, t] = 1.0

        return adj_out, adj_t, scores, torch.zeros(0, device=device)


class RelativeReweightPolicy(nn.Module):
    """Per-node relative reweighting via row-wise softmax.

    Unlike EdgeReweightPolicy (absolute multiplicative), this policy changes
    the *relative importance* among each node's neighbors. This is critical for
    DiffConv / row-normalized GNNs where T = D^{-1}A: uniform scaling cancels
    out after normalization, but per-node relative changes survive.

    Architecture:
      1. Node embedding: MLP(adj_row) -> h_i
      2. Edge score: s_ij = MLP([h_i; h_j; orig_w_ij])
      3. Per-node softmax: attn_ij = softmax_j(s_ij / tau) over neighbors of i
      4. Rescale: w_ij = attn_ij * orig_row_sum_i  (preserves degree)

    Identity init: all scores = 0 -> uniform softmax -> attn_ij = 1/deg(i)
    -> w_ij = row_sum/deg(i) = mean_neighbor_weight. Close to original when
    weights are uniform; for non-uniform weights we add a residual path.

    Args:
        num_nodes: number of spatial nodes.
        adj_dense: (N, N) original dense adjacency tensor.
        seq_len: input sequence length.
        hidden_dim: policy embedding dimension.
        temperature: softmax temperature.
    """

    def __init__(self, num_nodes, adj_dense, seq_len, hidden_dim=64,
                 temperature=1.0, num_new_edges=0, disable_reweight=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.temperature = temperature
        self.num_new_edges = num_new_edges
        self.disable_reweight = disable_reweight

        self.register_buffer('orig_adj', adj_dense.clone())
        row, col = torch.nonzero(adj_dense, as_tuple=True)
        self.register_buffer('edge_row', row)
        self.register_buffer('edge_col', col)
        self.register_buffer('orig_weights', adj_dense[row, col].clone())

        # Original row sums (degree-like) for rescaling
        self.register_buffer('orig_row_sum', adj_dense.sum(dim=1))

        # Base scores = log(orig_weight) so that softmax recovers original ratios
        # softmax(log(w_j)) = w_j / sum(w_k) -> original relative distribution
        self.register_buffer(
            'base_scores',
            torch.log(adj_dense[row, col].clamp(min=1e-8))
        )

        # Non-edge candidates for edge addition (upper-triangle, symmetrized in forward)
        if num_new_edges > 0:
            non_edge_mask = (adj_dense == 0)
            non_edge_mask.fill_diagonal_(False)
            idx = torch.arange(num_nodes)
            triu_mask = non_edge_mask & (idx.unsqueeze(1) < idx.unsqueeze(0))
            ne_row, ne_col = torch.nonzero(triu_mask, as_tuple=True)
            self.register_buffer('ne_row', ne_row)
            self.register_buffer('ne_col', ne_col)

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Score head: [h_i; h_j; orig_w] -> delta_score (residual on base)
        self.score_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Zero-init: delta=0 -> total_score = base_score -> identity
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

        # Edge addition head
        if num_new_edges > 0:
            self.add_head = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, adj_dense):
        N = self.num_nodes
        device = adj_dense.device
        h = self.node_encoder(adj_dense)

        if self.disable_reweight:
            adj_out = self.orig_adj.clone()
            delta = torch.zeros(self.edge_row.shape[0], device=device)
        else:
            h_i = h[self.edge_row]
            h_j = h[self.edge_col]
            edge_feat = torch.cat([h_i, h_j, self.orig_weights.unsqueeze(-1)], dim=-1)
            delta = self.score_head(edge_feat).squeeze(-1)

            total_scores = self.base_scores + delta

            score_mat = torch.full((N, N), -1e9, device=device, dtype=adj_dense.dtype)
            score_mat[self.edge_row, self.edge_col] = total_scores / self.temperature

            mask = (self.orig_adj > 0).float()
            attn = torch.softmax(score_mat, dim=1) * mask
            attn_sum = attn.sum(dim=1, keepdim=True).clamp(min=1e-8)
            attn = attn / attn_sum

            adj_out = attn * self.orig_row_sum.unsqueeze(1)

        # Edge addition: score non-edge candidates, top-k with symmetric insertion
        self._last_edges_added = 0
        self._last_new_edge_weights = None
        if self.num_new_edges > 0 and self.ne_row.shape[0] > 0:
            h_ne_i = h[self.ne_row]
            h_ne_j = h[self.ne_col]
            ne_scores = self.add_head(
                torch.cat([h_ne_i, h_ne_j], dim=-1)
            ).squeeze(-1)
            k = min(self.num_new_edges, ne_scores.shape[0])
            topk_vals, topk_idx = torch.topk(ne_scores, k)
            ne_weights = torch.sigmoid(topk_vals / self.temperature)
            avg_existing = adj_out[adj_out > 0].mean().detach()
            ne_weights = ne_weights * avg_existing
            sel_row = self.ne_row[topk_idx]
            sel_col = self.ne_col[topk_idx]
            adj_out[sel_row, sel_col] = ne_weights
            adj_out[sel_col, sel_row] = ne_weights
            self._last_edges_added = k * 2
            self._last_new_edge_weights = ne_weights.detach()

        # Temporal: causal baseline
        adj_t = torch.zeros(self.seq_len, self.seq_len, device=device)
        for t in range(self.seq_len - 1):
            adj_t[t + 1, t] = 1.0

        return adj_out, adj_t, delta, torch.zeros(0, device=device)
