"""AGCRN variant that combines adaptive adj with external (bilevel) adj.

The standard AGCRN computes adj = softmax(relu(E @ E^T)) from learned node
embeddings. This variant adds an external adjacency that can be set by
BilevelPredictor, combining via a learnable mixing weight:

    adj = alpha * adj_adaptive + (1-alpha) * ext_adj_normalized

When no external adj is set, falls back to standard AGCRN behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tsl.nn.layers.graph_convs import AdaptiveGraphConv
from tsl.nn.models.stgn.agcrn_model import AGCRNModel
from tsl.nn.utils import maybe_cat_exog


class AGCRNBilevelModel(AGCRNModel):
    r"""AGCRN with external adjacency support for bilevel graph optimization.

    Extends tsl's AGCRNModel to accept an external adjacency matrix that gets
    mixed with the internally learned adaptive graph.

    Additional args:
        mix_init (float): Initial mixing logit. sigmoid(0)=0.5 means equal
            weight to adaptive and external adj at init.
    """

    def __init__(self, input_size, output_size, horizon, n_nodes,
                 hidden_size=64, emb_size=10, exog_size=0, n_layers=1,
                 mix_init=0.0):
        super().__init__(input_size=input_size, output_size=output_size,
                         horizon=horizon, n_nodes=n_nodes,
                         hidden_size=hidden_size, emb_size=emb_size,
                         exog_size=exog_size, n_layers=n_layers)
        self._ext_adj = None
        self._mix_logit = nn.Parameter(torch.tensor(float(mix_init)))

    def set_ext_adj(self, adj):
        """Set external adjacency (called by BilevelPredictor)."""
        if adj is not None:
            row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
            self._ext_adj = adj / row_sum
        else:
            self._ext_adj = None

    def forward(self, x: Tensor, u=None) -> Tensor:
        x = maybe_cat_exog(x, u)
        x = self.input_encoder(x)
        emb = self.agrn.node_emb()
        adj_adaptive = AdaptiveGraphConv.compute_adj(emb)
        if self._ext_adj is not None:
            alpha = torch.sigmoid(self._mix_logit)
            ext = self._ext_adj.to(adj_adaptive.device)
            adj = alpha * adj_adaptive + (1.0 - alpha) * ext
        else:
            adj = adj_adaptive
        # Call RNNBase.forward (skip AGCRN.forward which recomputes adj)
        from tsl.nn.blocks.encoders.recurrent.base import RNNBase
        out = RNNBase.forward(self.agrn, x, adj=adj, e=emb)
        return self.readout(out)

