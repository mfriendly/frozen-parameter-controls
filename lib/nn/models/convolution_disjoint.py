from typing import Optional

import torch.nn.functional
from torch import nn
from tsl.nn import get_functional_activation
from tsl.nn.layers import TemporalConv

from .gnn_mixin import GCNMixin, GraphConvMixin, DiffConvMixin, ChebConvMixin, GATConvMixin
from .sparse_chebconv import SparseChebConv, SparseChebStack
from .stgnn_base import DisjointSTGNN
from ..layers import MPStack, CheckReceptiveField


class MPTCN(DisjointSTGNN):
    """The Message Passing Temporal Convolutional Network (MPTCN) model.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden features.
        exog_size (int, optional): Number of exogenous features. If :obj:`None`, no
            exogenous features are used.
            (default: :obj:`None`).
        output_size (int, optional): Number of output features. If :obj:`None`, it is
            set to :attr:`input_size`.
            (default: :obj:`None`).
        horizon (int, optional): Forecasting horizon. If :obj:`None`, the model adds a
            linear readout layer to map the hidden features to the output features.
            If not :obj:`None`, the model adds a linear readout layer that maps the
            hidden features to the output features for all time steps in the horizon.
            (default: :obj:`None`).
        n_layers (int, optional): Number of spatiotemporal message-passing layers.
            (default: :obj:`1`).
        tcn_layers (int, optional): Number of TCN layers in each spatiotemporal
            message-passing layer.
            (default: :obj:`1`).
        tcn_kwargs (dict, optional): Additional keyword arguments for the temporal
            convolutional layers.
            (default: :obj:`None`).
        gnn_layers (int, optional): Number of GNN layers in each spatiotemporal
            message-passing layer.
            (default: :obj:`1`).
        gnn_class (nn.Module, optional): Graph neural network class to use for the
            spatial message-passing. If :obj:`None`, it must be specified in a subclass.
            (default: :obj:`None`).
        gnn_kwargs (dict, optional): Additional keyword arguments for the graph neural
            network layers.
            (default: :obj:`None`).
        input_encoding (Optional[str], optional): Input encoding method. Can be one of
            :obj:`None` or :obj:`'identity'` for no encoding, :obj:`'orthogonal'` for
            an  orthogonal encoding with fixed weights, or :obj:`'linear'` for a
            trainable linear encoding.
            (default: :obj:`None`).
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 exog_size: int = None,
                 output_size: int = None,
                 horizon: int = None,
                 n_layers: int = 1,
                 tcn_layers: int = 1,
                 tcn_kwargs: dict = None,
                 gnn_layers: int = 1,
                 gnn_class: nn.Module = None,
                 gnn_kwargs: dict = None,
                 input_encoding: Optional[str] = None):
        # We use stacks of spatial convolutions, hence n_spatial_layers is always 1
        super(MPTCN, self).__init__(input_size=input_size,
                                    hidden_size=hidden_size,
                                    exog_size=exog_size,
                                    output_size=output_size,
                                    horizon=horizon,
                                    n_layers=n_layers,
                                    n_temporal_layers=tcn_layers,
                                    n_spatial_layers=int(gnn_layers > 0),
                                    input_encoding=input_encoding)

        # Temporal Convolutional Network ##############################################
        # Returns the encoded sequence of shape (batch, seq_len, n_nodes, hidden_size)
        tcn_default_kwargs = {
            "kernel_size": 3,
            "weight_norm": False,
            "dilation": 1,  # dilation=1 means no dilation
            "dilation_mod": None,  # modulo of dilation (None means infinite)
            "exponential_dilation": True,
            "activation": 'relu',
            "row_normalization": False,
        }
        tcn_default_kwargs.update(tcn_kwargs or {})

        d = dilation = tcn_default_kwargs.pop('dilation')
        exponential_dilation = tcn_default_kwargs.pop('exponential_dilation')
        dilation_mod = tcn_default_kwargs.pop('dilation_mod')

        tcn_activation = tcn_default_kwargs.pop('activation')
        self.tcn_activation = get_functional_activation(tcn_activation)

        self.row_normalization = tcn_default_kwargs.pop('row_normalization')
        if self.row_normalization:
            w = 1 / torch.arange(1, tcn_default_kwargs['kernel_size'] + 1,
                                 dtype=torch.float32)
            w = torch.flip(w, (0,)).view(1, 1, 1, -1).contiguous()
            self._norm_weights = nn.Parameter(w, requires_grad=False)

        self.tmps = nn.ModuleList()
        rf = 1
        for i in range(n_layers):
            self.tmps.append(nn.ModuleList())
            for l_t in range(tcn_layers):
                if i == 0 and l_t == 0:
                    in_channels = self.stmp_input_size
                else:
                    in_channels = hidden_size
                if exponential_dilation:
                    exp = l_t if dilation_mod is None else l_t % dilation_mod
                    d = dilation ** exp
                conv = TemporalConv(
                    input_channels=in_channels,
                    output_channels=hidden_size,
                    causal_pad=True,
                    dilation=d,
                    channel_last=False,
                    **tcn_default_kwargs,
                )
                self.tmps[i].append(conv)
                rf += d * (tcn_kwargs["kernel_size"] - 1)

        self.rf_temporal = rf
        self.check_temporal_rf = CheckReceptiveField(self.rf_temporal)

        # Graph Neural Network ########################################################
        self.gnn_class = gnn_class or self.gnn_class
        self.gnn_kwargs.update(gnn_kwargs or {})
        assert self.gnn_class is not None, 'gnn_class must be specified'

        mp_input_size = hidden_size if (tcn_layers * n_layers) else self.stmp_input_size
        self.gnns = nn.ModuleList([
            MPStack(gnn_class=self.gnn_class,
                    input_size=mp_input_size if i == 0 else hidden_size,
                    hidden_size=hidden_size, n_layers=gnn_layers, **self.gnn_kwargs)
            for i in range(n_layers)
        ])

    def tmp(self, x, l_out=None, l_in=None, temporal_adj=None):
        if l_in == 0:
            if l_out == 0:
                x = self.check_temporal_rf(x)
            x = torch.swapaxes(x, -3, -1)  # 'b t n f -> b f n t'
        if self.row_normalization:
            diff = x.size(-1) - self._norm_weights.size(-1)
            w = torch.nn.functional.pad(self._norm_weights, (diff, 0, 0, 0, 0, 0),
                                        mode='replicate')
            x = x * w
        z = self.tmps[l_out][l_in](x)
        z = self.tcn_activation(z)
        if l_in == self.n_temporal_layers - 1:
            z = torch.swapaxes(z, -3, -1)  # 'b f n t -> b t n f'
            if temporal_adj is not None:
                z = torch.einsum('btnf,ts->bsnf', z, temporal_adj)
            if l_out == self.n_layers - 1 and self.horizon is not None:
                z = z[:, -1]
        return z

    def smp(self, h, edge_index, edge_weight=None, l_out=None, l_in=None, **kwargs):
        h = self.gnns[l_out](h, edge_index, edge_weight, **kwargs)
        return h


class MPTCN_GCNModel(GCNMixin, MPTCN):
    pass


class MPTCN_GraphConvModel(GraphConvMixin, MPTCN):
    pass


class MPTCN_DiffConvModel(DiffConvMixin, MPTCN):
    pass


class MPTCN_ChebConvModel(ChebConvMixin, MPTCN):
    """MPTCN with SparseChebConv spatial layers (no per-timestep Python loop).

    After MPTCN.__init__ builds gnns as MPStack[PyG ChebConv], this replaces them
    with batched SparseChebConv layers.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 exog_size: int = None,
                 output_size: int = None,
                 horizon: int = None,
                 n_layers: int = 1,
                 tcn_layers: int = 1,
                 tcn_kwargs: dict = None,
                 gnn_layers: int = 1,
                 gnn_class: nn.Module = None,
                 gnn_kwargs: dict = None,
                 input_encoding: Optional[str] = None):
        super().__init__(
            input_size=input_size, hidden_size=hidden_size,
            exog_size=exog_size, output_size=output_size,
            horizon=horizon, n_layers=n_layers, tcn_layers=tcn_layers,
            tcn_kwargs=tcn_kwargs, gnn_layers=gnn_layers,
            gnn_class=gnn_class, gnn_kwargs=gnn_kwargs,
            input_encoding=input_encoding)
        K = self.gnn_kwargs.get('K', 3)
        bias = self.gnn_kwargs.get('bias', True)
        new_gnns = nn.ModuleList()
        for mpstack in self.gnns:
            in_ch = mpstack.input_size
            out_ch = mpstack.hidden_size
            n = mpstack.n_layers
            if n == 1:
                new_gnns.append(SparseChebConv(in_ch, out_ch, K=K, bias=bias))
            else:
                new_gnns.append(SparseChebStack(in_ch, out_ch, n_layers=n, K=K, bias=bias))
        self.gnns = new_gnns


class MPTCN_GATConvModel(GATConvMixin, MPTCN):
    pass
