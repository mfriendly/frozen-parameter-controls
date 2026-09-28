"""SparseChebConv: Batched Chebyshev graph convolution without per-timestep loops.

Drop-in replacement for PyG ChebConv that handles (B, T, N, F) inputs efficiently.

Two L̃ computation paths:
  - compute_scaled_laplacian_sparse: sparse CSR, fast for inference/inner steps
  - compute_scaled_laplacian_from_adj_s: dense, fully differentiable (bilevel outer step)

The forward() uses torch.einsum for dense L̃ (autograd-compatible) or torch.sparse.mm
for sparse L̃ (faster but no grad w.r.t. sparse values).
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


def compute_scaled_laplacian_sparse(edge_index: Tensor, edge_weight: Tensor,
                                     num_nodes: int, lambda_max: float = 2.0) -> Tensor:
    """Compute L̃ from edge_index/edge_weight → sparse CSR (fast, no grad through L̃).

    Use for vanilla training or inner steps where gradients through L̃ are not needed.
    Returns sparse CSR tensor.
    """
    row, col = edge_index[0], edge_index[1]

    D = torch.zeros(num_nodes, device=edge_index.device)
    D.scatter_add_(0, row, edge_weight)

    D_inv_sqrt = torch.zeros_like(D)
    mask = D > 0
    D_inv_sqrt[mask] = D[mask].pow(-0.5)

    norm_weights = D_inv_sqrt[row] * edge_weight * D_inv_sqrt[col]

    scale = 2.0 / lambda_max
    off_diag_vals = -scale * norm_weights

    diag_vals = torch.full((num_nodes,), scale - 1.0, device=edge_index.device)
    self_loop_mask = row == col
    if self_loop_mask.any():
        diag_vals.scatter_add_(0, row[self_loop_mask],
                               -scale * norm_weights[self_loop_mask])
        non_self = ~self_loop_mask
        row, col = row[non_self], col[non_self]
        off_diag_vals = off_diag_vals[non_self]

    diag_idx = torch.arange(num_nodes, device=edge_index.device)
    all_row = torch.cat([diag_idx, row])
    all_col = torch.cat([diag_idx, col])
    all_val = torch.cat([diag_vals, off_diag_vals])

    return torch.sparse_coo_tensor(
        torch.stack([all_row, all_col]), all_val, size=(num_nodes, num_nodes)
    ).coalesce().to_sparse_csr()


def compute_scaled_laplacian_from_adj_s(adj_s: Tensor,
                                         lambda_max: float = 2.0) -> Tensor:
    """Compute dense L̃ from dense adjacency matrix. Fully differentiable.

    Use for bilevel outer step: gradients flow adj_s → L̃ → ChebConv output → val_loss.

    L̃ = (2/λ_max) * L_norm - I  where  L_norm = I - D^{-1/2} A D^{-1/2}
    With λ_max=2.0 (standard approx): L̃ = -D^{-1/2} A D^{-1/2}

    Args:
        adj_s: (N, N) dense adjacency, possibly with grad.
        lambda_max: scaling factor (default 2.0 to skip eigendecomp).

    Returns:
        L_tilde: (N, N) dense tensor.
    """
    N = adj_s.shape[0]
    D = adj_s.sum(dim=1)
    D_inv_sqrt = D.clamp(min=1e-8).pow(-0.5)
    A_norm = D_inv_sqrt.unsqueeze(1) * adj_s * D_inv_sqrt.unsqueeze(0)
    eye = torch.eye(N, device=adj_s.device, dtype=adj_s.dtype)
    L_norm = eye - A_norm
    return (2.0 / lambda_max) * L_norm - eye


class SparseChebConv(nn.Module):
    """Chebyshev spectral graph convolution.

    Handles arbitrary leading dimensions (..., N, F) without per-timestep Python loops.

    For dense L̃:  uses torch.einsum → differentiable, CUDA-optimized (cuBLAS).
    For sparse L̃: uses torch.sparse.mm → faster but no grad w.r.t. L̃ values.

    Chebyshev recurrence:
        T_0(L̃)x = x
        T_1(L̃)x = L̃x
        T_k(L̃)x = 2·L̃·T_{k-1}(L̃)x - T_{k-2}(L̃)x

    Output: Σ_{k=0}^{K-1} W_k · T_k(L̃)x  + bias
    """

    def __init__(self, in_channels: int, out_channels: int, K: int = 3,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.lins = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False) for _ in range(K)
        ])
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            nn.init.xavier_uniform_(lin.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, L_tilde: Tensor) -> Tensor:
        """
        Args:
            x: (..., N, F_in) with arbitrary leading dims.
            L_tilde: (N, N) dense or sparse scaled Laplacian.

        Returns:
            out: (..., N, F_out).
        """
        is_sparse = L_tilde.is_sparse or L_tilde.layout == torch.sparse_csr

        if is_sparse:
            return self._forward_sparse(x, L_tilde)
        else:
            return self._forward_dense(x, L_tilde)

    def _forward_dense(self, x: Tensor, L_tilde: Tensor) -> Tensor:
        """Dense path: einsum over (..., N, F). Supports autograd through L̃."""
        # T_0 = I  →  T_0(L̃)x = x
        T_prev = x
        out = self.lins[0](x)

        if self.K > 1:
            # T_1 = L̃  →  T_1(L̃)x = L̃x
            # einsum: L[m,n] * x[...,n,f] → y[...,m,f]
            T_curr = torch.einsum('mn,...nf->...mf', L_tilde, x)
            out = out + self.lins[1](T_curr)

            for k in range(2, self.K):
                T_next = 2.0 * torch.einsum('mn,...nf->...mf', L_tilde, T_curr) - T_prev
                out = out + self.lins[k](T_next)
                T_prev = T_curr
                T_curr = T_next

        if self.bias is not None:
            out = out + self.bias
        return out

    def _forward_sparse(self, x: Tensor, L_tilde: Tensor) -> Tensor:
        """Sparse path: single reshape + sparse matmul. No grad through L̃."""
        leading = x.shape[:-2]
        N, F_in = x.shape[-2], x.shape[-1]
        BT = math.prod(leading) if leading else 1

        x_3d = x.reshape(BT, N, F_in)
        # (N, BT*F) layout for sparse mm
        x_2d = x_3d.permute(1, 0, 2).reshape(N, BT * F_in)

        T_prev_2d = x_2d
        out = self.lins[0](x_3d)

        if self.K > 1:
            T_curr_2d = torch.sparse.mm(L_tilde, x_2d)
            T_curr_3d = T_curr_2d.reshape(N, BT, F_in).permute(1, 0, 2)
            out = out + self.lins[1](T_curr_3d)

            for k in range(2, self.K):
                T_next_2d = 2.0 * torch.sparse.mm(L_tilde, T_curr_2d) - T_prev_2d
                T_next_3d = T_next_2d.reshape(N, BT, F_in).permute(1, 0, 2)
                out = out + self.lins[k](T_next_3d)
                T_prev_2d = T_curr_2d
                T_curr_2d = T_next_2d

        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*leading, N, self.out_channels)


class SparseChebStack(nn.Module):
    """Stack of SparseChebConv layers — replaces MPStack for ChebConv.

    forward(x, L_tilde) applies n_layers of SparseChebConv with ReLU activations
    (matching MPStack's behaviour with activation=None → identity in MPTCN default).
    """

    def __init__(self, input_size: int, hidden_size: int,
                 n_layers: int = 1, K: int = 3, bias: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            SparseChebConv(input_size if i == 0 else hidden_size,
                           hidden_size, K=K, bias=bias)
            for i in range(n_layers)
        ])

    def forward(self, x: Tensor, L_tilde: Tensor) -> Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x, L_tilde))
        return x


if __name__ == '__main__':
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    N, F_in, F_out, K = 307, 64, 64, 3
    B, T = 64, 12

    num_edges = 268 * 2
    src = torch.randint(0, N, (num_edges,))
    dst = torch.randint(0, N, (num_edges,))
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edge_index = torch.stack([src, dst]).to(device)
    edge_weight = torch.rand(edge_index.shape[1], device=device)

    adj_dense = torch.zeros(N, N, device=device)
    adj_dense[edge_index[0], edge_index[1]] = edge_weight
    L_tilde_dense = compute_scaled_laplacian_from_adj_s(adj_dense)
    L_tilde_sparse = compute_scaled_laplacian_sparse(edge_index, edge_weight, N)

    conv = SparseChebConv(F_in, F_out, K=K).to(device)
    x = torch.randn(B, T, N, F_in, device=device)

    import time
    torch.cuda.synchronize() if device == 'cuda' else None
    t0 = time.time()
    for _ in range(10):
        out_dense = conv(x, L_tilde_dense)
    torch.cuda.synchronize() if device == 'cuda' else None
    t1 = time.time()
    print(f"Dense  10x: {(t1-t0)*1000:.1f}ms  ({(t1-t0)/10*1000:.1f}ms/fwd)")

    torch.cuda.synchronize() if device == 'cuda' else None
    t0 = time.time()
    for _ in range(10):
        out_sparse = conv(x, L_tilde_sparse)
    torch.cuda.synchronize() if device == 'cuda' else None
    t1 = time.time()
    print(f"Sparse 10x: {(t1-t0)*1000:.1f}ms  ({(t1-t0)/10*1000:.1f}ms/fwd)")

    x.requires_grad_(True)
    adj_dense.requires_grad_(True)
    L_td = compute_scaled_laplacian_from_adj_s(adj_dense)
    out = conv(x, L_td)
    out.sum().backward()
    print(f"Grad x: {x.grad is not None}")
    print(f"Grad adj: {adj_dense.grad is not None}")
