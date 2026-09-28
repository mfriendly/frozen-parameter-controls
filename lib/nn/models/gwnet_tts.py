from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.typing import Adj, OptTensor
from tsl.nn.models import GraphWaveNetModel
from tsl.nn.utils import maybe_cat_exog

from lib.nn.layers import OrthogonalEncoder


class GraphWaveNetTTSModel(GraphWaveNetModel):
    r"""The time-then-space version of Graph WaveNet model from the paper
    `"Graph WaveNet for Deep Spatial-Temporal Graph Modeling"
    <https://arxiv.org/abs/1906.00121>`_ (Wu et al., IJCAI 2019).
    """
    supports_temporal_adj = True

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 horizon: int,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 ff_size: int = 256,
                 n_layers: int = 8,
                 gnn_layers: Optional[int] = None,
                 temporal_kernel_size: int = 2,
                 spatial_kernel_size: int = 2,
                 learned_adjacency: bool = True,
                 n_nodes: Optional[int] = None,
                 emb_size: int = 10,
                 dilation: int = 2,
                 dilation_mod: int = 2,
                 norm: str = 'batch',
                 dropout: float = 0.3):
        super(GraphWaveNetTTSModel, self).__init__(
            input_size=input_size,
            output_size=output_size,
            horizon=horizon,
            exog_size=exog_size,
            hidden_size=hidden_size,
            ff_size=ff_size,
            n_layers=n_layers,
            temporal_kernel_size=temporal_kernel_size,
            spatial_kernel_size=spatial_kernel_size,
            learned_adjacency=learned_adjacency,
            n_nodes=n_nodes,
            emb_size=emb_size,
            dilation=dilation,
            dilation_mod=dilation_mod,
            norm=norm,
            dropout=dropout)
        if gnn_layers is not None:
            assert gnn_layers <= n_layers, \
                "GNN layers must be less than or equal to total layers"
            self.sconvs = self.sconvs[:gnn_layers]
            self.dense_sconvs = self.dense_sconvs[:gnn_layers]
        self.orthogonal_proj = OrthogonalEncoder(hidden_size, ff_size)

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None,
                u: OptTensor = None,
                temporal_adj: OptTensor = None) -> Tensor:
        """"""
        # x: [b t n f]
        x = maybe_cat_exog(x, u)

        if self.receptive_field > x.size(1):
            x = F.pad(x, (0, 0, 0, 0, self.receptive_field - x.size(1), 0))

        if len(self.dense_sconvs):
            adj_z = self.get_learned_adj()

        x = self.input_encoder(x)

        out = torch.zeros(1, x.size(1), 1, 1, device=x.device)

        # Temporal processing
        for i, (tconv, skip_conn, norm) in enumerate(zip(
                self.tconvs, self.skip_connections, self.norms)):
            res = x
            x = tconv(x)
            out = skip_conn(x) + out[:, -x.size(1):]
            x = self.dropout(x)
            x = x + res[:, -x.size(1):]
            x = norm(x)

        # Temporal rewiring: learned temporal mixing after dilated conv
        if temporal_adj is not None:
            x = torch.einsum('btnf,ts->bsnf', x, temporal_adj)

        # Spatial processing
        for i, sconv in enumerate(self.sconvs):
            xs = sconv(x, edge_index, edge_weight)
            if len(self.dense_sconvs):
                x = xs + self.dense_sconvs[i](x, adj_z)
            else:
                x = xs
        out = self.orthogonal_proj(x) + out[:, -x.size(1):]

        return self.readout(out)
