"""MPGRU: Decoupled DiffConv + GRU backbone.

Same decoupled spatial-temporal architecture as MPTCN, but replaces TCN temporal
module with standard nn.GRU. DiffConv is applied independently from GRU — the
GRU never sees the adjacency matrix.

Architecture per ST layer:
    z = GRU(x)          # temporal: per-node sequence processing
    h = DiffConv(z, A)  # spatial: graph convolution per timestep

This contrasts with DCRNN where DiffConv is coupled INSIDE the GRU cell gates.
"""
from typing import Optional

import torch
from torch import nn, Tensor

from .gnn_mixin import DiffConvMixin
from .stgnn_base import DisjointSTGNN
from ..layers import MPStack


class MPGRU(DisjointSTGNN):
    """Message Passing GRU: decoupled DiffConv (spatial) + GRU (temporal).

    Mirrors MPTCN exactly except temporal TCN layers are replaced by GRU layers.
    The GRU processes each node's temporal sequence independently.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 exog_size: int = None,
                 output_size: int = None,
                 horizon: int = None,
                 n_layers: int = 1,
                 gru_layers: int = 1,
                 gnn_layers: int = 1,
                 gnn_class: nn.Module = None,
                 gnn_kwargs: dict = None,
                 dropout: float = 0.0,
                 input_encoding: Optional[str] = None):
        super().__init__(input_size=input_size,
                         hidden_size=hidden_size,
                         exog_size=exog_size,
                         output_size=output_size,
                         horizon=horizon,
                         n_layers=n_layers,
                         n_temporal_layers=1,
                         n_spatial_layers=int(gnn_layers > 0),
                         input_encoding=input_encoding)

        self._gru_layers = gru_layers

        # GRU temporal modules — one per ST layer
        self.grus = nn.ModuleList()
        for i in range(n_layers):
            in_ch = self.stmp_input_size if i == 0 else hidden_size
            self.grus.append(nn.GRU(
                input_size=in_ch,
                hidden_size=hidden_size,
                num_layers=gru_layers,
                batch_first=True,
                dropout=dropout if gru_layers > 1 else 0.0,
            ))

        # Spatial GNN modules — same as MPTCN
        self.gnn_class = gnn_class or self.gnn_class
        self.gnn_kwargs.update(gnn_kwargs or {})
        assert self.gnn_class is not None, 'gnn_class must be specified'

        self.gnns = nn.ModuleList([
            MPStack(gnn_class=self.gnn_class,
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    n_layers=gnn_layers,
                    **self.gnn_kwargs)
            for _ in range(n_layers)
        ])

    def tmp(self, x: Tensor, l_out: int = None, l_in: int = None,
            temporal_adj=None) -> Tensor:
        """GRU temporal processing. x: (B, T, N, F) → (B, T, N, H).

        Reshapes to (B*N, T, F) for per-node GRU, then back.
        """
        B, T, N, F = x.shape
        # (B, T, N, F) → (B*N, T, F) — each node is an independent sequence
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        # cuDNN RNN backward requires training mode; bilevel outer step runs
        # in eval mode but still needs gradients through the GRU.
        gru = self.grus[l_out]
        force_train = not gru.training and torch.is_grad_enabled()
        if force_train:
            gru.train()
        out, _ = gru(x_flat)  # (B*N, T, H)
        if force_train:
            gru.eval()
        out = out.reshape(B, N, T, -1).permute(0, 2, 1, 3)  # (B, T, N, H)

        if temporal_adj is not None:
            out = torch.einsum('btnf,ts->bsnf', out, temporal_adj)
        if l_out == self.n_layers - 1 and self.horizon is not None:
            out = out[:, -1]  # (B, N, H) — last timestep for readout
        return out

    def smp(self, h, edge_index, edge_weight=None, l_out=None, l_in=None, **kwargs):
        h = self.gnns[l_out](h, edge_index, edge_weight, **kwargs)
        return h


class MPGRU_DiffConvModel(DiffConvMixin, MPGRU):
    """Decoupled DiffConv + GRU model."""
    pass
