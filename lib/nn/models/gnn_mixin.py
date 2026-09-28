import torch
from torch.nn import Module
from torch_geometric.nn import GCNConv, GATConv, ChebConv
from tsl.nn.layers import GraphConv, DiffConv
from .sparse_chebconv import compute_scaled_laplacian_from_adj_s


class GCNMixin:
    gnn_class: Module = GCNConv
    gnn_kwargs: dict = {
        'add_self_loops': True,
        'normalize': True,
        'bias': True,
        'cached': True,
    }


class GraphConvMixin:
    gnn_class: Module = GraphConv
    gnn_kwargs: dict = {
        'root_weight': True,
        'norm': 'asym',
        'activation': 'relu',
        'cached': True,
    }


class DiffConvMixin:
    gnn_class: Module = DiffConv
    gnn_kwargs: dict = {
        'k': 2,
        'root_weight': True,
        'add_backward': True,
        'activation': 'relu',
    }

    def smp(self, h, edge_index, edge_weight=None, l_out=None, l_in=None, **kwargs):
        return super().smp(h, edge_index, edge_weight, l_out=l_out, l_in=l_in,
                           cache_support=True, **kwargs)


class ChebConvMixin:
    # gnn_class kept as ChebConv so MPTCN.__init__ can build gnns normally;
    # MPTCN_ChebConvModel.__init__ replaces self.gnns with SparseChebConv afterwards.
    gnn_class: Module = ChebConv
    gnn_kwargs: dict = {
        'K': 3,
        'normalization': 'sym',
        'bias': True,
    }

    def smp(self, h, edge_index, edge_weight=None, l_out=None, l_in=None, **kwargs):
        """Call SparseChebConv with dense L̃ — no per-timestep Python loop.

        Extracts edge weights from SparseTensor or COO, reconstructs dense
        adjacency, computes L̃ once, then calls self.gnns[l_out](h, L_tilde).
        The dense path supports autograd for the bilevel outer step.
        """
        N = h.shape[-2]

        if hasattr(edge_index, 'sparse_sizes'):
            row, col, val = edge_index.coo()
            ei = torch.stack([row, col], dim=0)
            ew = val
        else:
            ei = edge_index
            ew = edge_weight

        # Build dense adj: sparse_coo_tensor → to_dense() is differentiable w.r.t. ew
        adj_dense = torch.sparse_coo_tensor(
            ei, ew, size=(N, N), device=h.device, dtype=h.dtype
        ).to_dense()
        L_tilde = compute_scaled_laplacian_from_adj_s(adj_dense)

        return self.gnns[l_out](h, L_tilde)


class GATConvMixin:
    gnn_class: Module = GATConv
    gnn_kwargs: dict = {
        'heads': 4,
        'concat': False,
        'add_self_loops': True,
        'negative_slope': 0.2,
        'dropout': 0.0,
    }

    def smp(self, h, edge_index, edge_weight=None, l_out=None, l_in=None, **kwargs):
        # GATConv expects 2D (N, F); loop over leading dims (B, T, ...)
        leading = h.shape[:-2]
        N, F = h.shape[-2], h.shape[-1]
        h_flat = h.reshape(-1, N, F)

        # Convert SparseTensor -> COO + edge_attr for GATConv
        if hasattr(edge_index, 'sparse_sizes'):
            row, col, val = edge_index.coo()
            ei = torch.stack([row, col], dim=0)
            ea = val.unsqueeze(-1) if val is not None else None
        else:
            ei = edge_index
            ea = edge_weight.unsqueeze(-1) if edge_weight is not None else None

        outs = []
        for t in range(h_flat.shape[0]):
            outs.append(self.gnns[l_out](h_flat[t], ei, ea))
        h_out = torch.stack(outs, dim=0)
        return h_out.reshape(*leading, N, -1)
