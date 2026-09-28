"""Bottleneck and oversquashing analysis utilities.

Provides spectral gap computation, spatial/temporal bottleneck scores,
and before/after rewiring comparison metrics.
"""
import numpy as np
import torch
import scipy.sparse as sp


def compute_spectral_gap(adj):
    """Compute spectral gap (lambda_2) of the normalized graph Laplacian.

    Args:
        adj: adjacency matrix — torch dense (N,N), scipy sparse, or numpy (N,N).

    Returns:
        float: second smallest eigenvalue of L_norm = I - D^{-1/2} A D^{-1/2}.
    """
    if isinstance(adj, torch.Tensor):
        A = adj.float()
        d = A.sum(dim=1)
        d_inv_sqrt = (d + 1e-8).pow(-0.5)
        L_norm = (torch.eye(A.size(0), device=A.device)
                  - d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0))
        eigvals = torch.linalg.eigvalsh(L_norm)
        return eigvals[1].item()

    if sp.issparse(adj):
        adj = adj.toarray()
    A = np.array(adj, dtype=np.float64)
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, d ** -0.5, 0.0)
    L_norm = np.eye(A.shape[0]) - d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    eigvals = np.linalg.eigvalsh(L_norm)
    return float(eigvals[1])


def compute_effective_resistance(adj):
    """Compute total effective resistance of the graph.

    Total effective resistance = n * trace(L^+) where L^+ is the
    pseudoinverse of the Laplacian.

    Args:
        adj: adjacency matrix (torch dense or numpy).

    Returns:
        float: total effective resistance.
    """
    if isinstance(adj, torch.Tensor):
        A = adj.float()
        d = A.sum(dim=1)
        L = torch.diag(d) - A
        pinv = torch.linalg.pinv(L, hermitian=True)
        return (A.size(0) * torch.trace(pinv)).item()

    A = np.array(adj, dtype=np.float64)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    pinv = np.linalg.pinv(L)
    return float(A.shape[0] * np.trace(pinv))


def spatial_bottleneck_score(adj, num_layers):
    """Compute spatial bottleneck score matrix.

    Score(i,j) = -log(A_norm^L [i,j]) measures how much information
    is lost from node j to node i after L message-passing layers.

    Args:
        adj: (N, N) adjacency tensor.
        num_layers: number of GNN layers.

    Returns:
        (N, N) bottleneck score matrix.
    """
    if not isinstance(adj, torch.Tensor):
        adj = torch.tensor(adj, dtype=torch.float32)
    A = adj.float()
    d = A.sum(dim=1)
    d_inv_sqrt = (d + 1e-8).pow(-0.5)
    A_norm = d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)
    A_k = torch.matrix_power(A_norm, num_layers)
    return -torch.log(A_k + 1e-12)


def temporal_bottleneck_score(seq_len, num_layers, dilation=1):
    """Compute temporal bottleneck score matrix.

    Models information flow in causal temporal convolutions.

    Args:
        seq_len: length of temporal sequence.
        num_layers: number of temporal conv layers.
        dilation: dilation factor.

    Returns:
        (T, T) temporal bottleneck score matrix.
    """
    T_adj = torch.zeros(seq_len, seq_len)
    for t in range(seq_len - 1):
        T_adj[t + 1, t] = 1.0
    for t in range(seq_len):
        for d in range(1, dilation + 1):
            if t - d >= 0:
                T_adj[t, t - d] = 1.0
    d_t = T_adj.sum(dim=1)
    d_inv = (d_t + 1e-8).pow(-1)
    T_norm = d_inv.unsqueeze(1) * T_adj
    T_k = torch.matrix_power(T_norm, num_layers)
    return -torch.log(T_k + 1e-12)


def bottleneck_improvement(adj_orig, adj_rewired, num_layers):
    """Compare bottleneck metrics before and after rewiring.

    Args:
        adj_orig: original adjacency.
        adj_rewired: rewired adjacency.
        num_layers: number of GNN layers.

    Returns:
        dict with comparison metrics.
    """
    gap_orig = compute_spectral_gap(adj_orig)
    gap_rewired = compute_spectral_gap(adj_rewired)

    res_orig = compute_effective_resistance(adj_orig)
    res_rewired = compute_effective_resistance(adj_rewired)

    score_orig = spatial_bottleneck_score(adj_orig, num_layers)
    score_rewired = spatial_bottleneck_score(adj_rewired, num_layers)

    return {
        'spectral_gap_orig': gap_orig,
        'spectral_gap_rewired': gap_rewired,
        'spectral_gap_improvement': gap_rewired - gap_orig,
        'total_resistance_orig': res_orig,
        'total_resistance_rewired': res_rewired,
        'total_resistance_reduction': res_orig - res_rewired,
        'mean_bottleneck_orig': score_orig.mean().item(),
        'mean_bottleneck_rewired': score_rewired.mean().item(),
        'mean_bottleneck_reduction': score_orig.mean().item() - score_rewired.mean().item(),
    }
