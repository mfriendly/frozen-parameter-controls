from typing import Optional

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.autograd.functional import jacobian
from torch_geometric.typing import Adj
from tsl.nn import maybe_cat_exog
from tsl.nn.models import BaseModel

from lib.nn.layers import OrthogonalEncoder


class STGNN(BaseModel):
    """Base class for SpatioTemporal Graph Neural Networks (STGNNs).

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
        input_encoding (Optional[str], optional): Input encoding method. Can be one of
            :obj:`None` or :obj:`'identity'` for no encoding, :obj:`'orthogonal'` for
            an  orthogonal encoding with fixed weights, or :obj:`'linear'` for a
            trainable linear encoding.
            (default: :obj:`None`).
    """
    gnn_class: nn.Module = None
    gnn_kwargs: dict = dict()
    supports_temporal_adj: bool = True

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 exog_size: int = None,
                 output_size: int = None,
                 horizon: int = None,
                 n_layers: int = 1,
                 input_encoding: Optional[str] = None):
        super(STGNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.exog_size = exog_size or 0
        self.output_size = output_size or input_size
        self.horizon = horizon
        self.n_layers = n_layers
        self.input_encoding = input_encoding

        # Encoder #############################################################
        input_size = self.input_size + self.exog_size
        if input_encoding is None or input_encoding == 'identity':
            self.upscale = nn.Identity()
        elif input_encoding == 'orthogonal':
            assert input_size < hidden_size, 'input_size must be less than hidden_size'
            self.upscale = OrthogonalEncoder(input_size, hidden_size)
        elif input_encoding == 'linear':
            self.upscale = nn.Linear(input_size, hidden_size)
        else:
            raise ValueError(f'Unknown input_encoding: {input_encoding}')

        # Readout #############################################################
        if horizon is not None:
            self.decoder = nn.Sequential(
                nn.Linear(self.hidden_size, self.output_size * horizon),
                Rearrange('... n (h f) -> ... h n f', h=horizon)
            )
        else:
            self.decoder = nn.Linear(self.hidden_size, self.output_size)

    @property
    def stmp_input_size(self):
        """Input size for the spatiotemporal message-passing layers."""
        if self.input_encoding is None or self.input_encoding == 'identity':
            return self.input_size + self.exog_size
        return self.hidden_size

    def encoder(self, x: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """The encoder module.

        Args:
            x (Tensor): Input tensor of shape ``[B, T, N, d_in]``, where
                :math:`B` is the batch size, :math:`T` is the number of time steps,
                :math:`N` is the number of nodes, and :math:`d_{in}` is the number of
                input features.
            u (Tensor, optional): Exogenous input tensor of shape ``[B, T, N, d_u]`` or
                ``[B, T, d_u]``, where :math:`d_u` is the number of exogenous features.
                (default: :obj:`None`).

        Returns:
            Tensor: Encoded tensor of shape ``[B, T, N, d_h]``, where :math:`d_h` is
                the number of hidden features.
        """
        x = maybe_cat_exog(x, u)
        return self.upscale(x)

    def stmp(self, x: Tensor, edge_index: Adj,
             edge_weight: Optional[Tensor] = None,
             *,
             layer: int = None,
             **kwargs) -> Tensor:
        """The spatiotemporal message-passing (STMP) layer.

        Args:
            x (Tensor): Input tensor of shape ``[B, T, N, d_h]``, where
                :math:`B` is the batch size, :math:`T` is the number of time steps,
                :math:`N` is the number of nodes, and :math:`d_h` is the number of
                hidden features.
            edge_index (Adj): Connectivity information of the graph in the form of
                sparse adjacency matrix or edge index.
            edge_weight (Tensor, optional): Edge weights in case of a weighted graph
                represented as an edge index.
                (default: :obj:`None`).
            layer (int, optional): Layer index.
            **kwargs: Additional arguments for the spatiotemporal message-passing layer.

        Returns:
            Tensor: Output tensor of shape ``[B, T, N, d_h]``.
        """
        raise NotImplementedError

    def readout(self, h: Tensor) -> Tensor:
        """The readout module.

        Args:
            h (Tensor): Input tensor of shape ``[B, T, N, d_h]``, where
                :math:`B` is the batch size, :math:`T` is the number of time steps,
                :math:`N` is the number of nodes, and :math:`d_h` is the number of
                hidden features.

        Returns:
            Tensor: Output tensor of shape ``[B, T, N, d_out]`` if horizon is not
                :obj:`None`, or ``[B, H, N, d_out]`` otherwise, where :math:`d_out` is
                the number of output features and :math:`H` is the forecasting horizon.
        """
        return self.decoder(h)

    def forward(self, x: Tensor,
                edge_index: Adj,
                edge_weight: Optional[Tensor] = None,
                u: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        """The forward pass of the STGNN model.

        Args:
            x (Tensor): Input tensor of shape ``[B, T, N, d_in]``, where
                :math:`B` is the batch size, :math:`T` is the number of time steps,
                :math:`N` is the number of nodes, and :math:`d_{in}` is the number of
                input features.
            edge_index (Adj): Connectivity information of the graph in the form of
                sparse adjacency matrix or edge index.
            edge_weight (Tensor, optional): Edge weights in case of a weighted graph
                represented as an edge index.
                (default: :obj:`None`).
            u (Tensor, optional): Exogenous input tensor of shape ``[B, T, N, d_u]`` or
                ``[B, T, d_u]``, where :math:`d_u` is the number of exogenous features.
                (default: :obj:`None`).
            **kwargs: Additional arguments for the spatiotemporal message-passing
                layers.
        """
        # x: [batches steps nodes features]
        # ENCODER   ###########################################################
        h = self.encoder(x, u)

        # SPATIOTEMPORAL MESSAGE-PASSING   ####################################
        for layer in range(self.n_layers):
            h = self.stmp(h, edge_index, edge_weight, **kwargs,
                          layer=layer)

        # DECODER   ###########################################################
        out = self.readout(h)

        return out

    def jacobian(self, x: Tensor, edge_index: Adj,
                 edge_weight: Optional[Tensor] = None,
                 u: Optional[Tensor] = None,
                 stmp_only: bool = True,
                 **kwargs) -> Tensor:
        r"""Compute the jacobian of the forward pass with respect to the input.

        Args:
            x (Tensor): Input tensor of shape ``[B, T, N, d_in]``, where
                :math:`B` is the batch size, :math:`T` is the number of time steps,
                :math:`N` is the number of nodes, and :math:`d_{in}` is the number of
                input features.
            edge_index (Adj): Connectivity information of the graph in the form of
                sparse adjacency matrix or edge index.
            edge_weight (Tensor, optional): Edge weights in case of a weighted graph
                represented as an edge index.
                (default: :obj:`None`).
            u (Tensor, optional): Exogenous input tensor of shape ``[B, T, N, d_u]`` or
                ``[B, T, d_u]``, where :math:`d_u` is the number of exogenous features.
                (default: :obj:`None`).
            stmp_only (bool, optional): Whether to compute the jacobian
                of the spatiotemporal message-passing only or of the full forward pass.
                (default: :obj:`True`).
            **kwargs: Additional arguments for the spatiotemporal message-passing
                layers.

        Returns:
            Tensor: Jacobian of the forward pass with respect to the input. The
                Jacobian sizes are :math:`\mathbb{R}^{d \times d}` if :attr:`stmp_only`
                is :obj:`True`, and :math:`\mathbb{R}^{d_x \times d_x}` otherwise.
        """
        # x: [batches steps nodes features]
        kwargs.update({'edge_index': edge_index, 'edge_weight': edge_weight})

        if stmp_only:
            with torch.no_grad():
                x = self.encoder(x, u)

            def fwd(h):
                # SPATIOTEMPORAL MESSAGE-PASSING   ############################
                for layer in range(self.n_layers):
                    h = self.stmp(h, **kwargs, layer=layer)
                return h
        else:
            x = maybe_cat_exog(x, u)

            def fwd(fwd_x):
                return self(x=fwd_x, **kwargs)

        # Compute the jacobian of the forward pass by looping over the batch
        jac_x = []
        for i in range(x.size(0)):
            jac_xi = jacobian(fwd, x[i:i + 1])  # [b=1,(h,) n, d_o, b=1, t, n, d_i]
            jac_x.append(jac_xi.detach())
        jac_x = torch.cat(jac_x, dim=0)  # shape: [b,(h,) n, d_o, b=1, t, n, d_i]

        # Compute the jacobian of the forward pass by vectorizing the batch
        # shape: [b,(h,) n, d_o, b=1, t, n, d_i]
        # jac_x = jacobian(fwd, x, vectorize=True)
        # jac_x_ = torch.vmap(torch.func.jacrev(fwd))(x[:, None])[:, 0]

        # Reshape to: [b, t, n, n, d_o, d_i]
        jac_x = rearrange(jac_x, "b ... n d 1 t m f -> b ... t n m d f")

        return jac_x


class DisjointSTGNN(STGNN):
    """Base class for Disjoint SpatioTemporal Graph Neural Networks (Disjoint STGNNs).

    In Disjoint STGNNs, the spatiotemporal message-passing layer is decomposed into
    a stack of temporal message-passing layers and a stack of spatial message-passing
    layers.

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
        n_temporal_layers (int, optional): Number of temporal message-passing layers
            in each spatiotemporal message-passing layer.
            (default: :obj:`1`).
        n_spatial_layers (int, optional): Number of spatial message-passing layers
            in each spatiotemporal message-passing layer.
            (default: :obj:`1`).
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
                 n_temporal_layers: int = 1,
                 n_spatial_layers: int = 1,
                 input_encoding: Optional[str] = None):
        super(DisjointSTGNN, self).__init__(input_size=input_size,
                                            hidden_size=hidden_size,
                                            exog_size=exog_size,
                                            output_size=output_size,
                                            horizon=horizon,
                                            n_layers=n_layers,
                                            input_encoding=input_encoding)
        self.n_temporal_layers = n_temporal_layers
        self.n_spatial_layers = n_spatial_layers

    def tmp(self, x: Tensor,
            *,
            l_out: int = None,
            l_in: int = None,
            temporal_adj: Optional[Tensor] = None) -> Tensor:
        """The temporal message-passing (TMP) layer.

        Args:
            x (Tensor): Input tensor of shape ``[B, T, N, d_h]``, where
                :math:`B` is the batch size, :math:`T` is the number of time steps,
                :math:`N` is the number of nodes, and :math:`d_h` is the number of
                hidden features.
            l_out (int, optional): Output layer index.
            l_in (int, optional): Input layer index.
            temporal_adj (Tensor, optional): Learned temporal adjacency of shape
                ``[T, T]``. If :obj:`None`, standard temporal processing is used.
        """
        raise NotImplementedError

    def smp(self, x: Tensor, edge_index: Adj,
            edge_weight: Optional[Tensor] = None,
            *,
            l_out: int = None,
            l_in: int = None,
            **kwargs) -> Tensor:
        """The spatial message-passing (SMP) layer.

        Args:
            x (Tensor): Input tensor of shape ``[B, T, N, d_h]``, where
                :math:`B` is the batch size, :math:`T` is the number of time steps,
                :math:`N` is the number of nodes, and :math:`d_h` is the number of
                hidden features.
            edge_index (Adj): Connectivity information of the graph in the form of
                sparse adjacency matrix or edge index.
            edge_weight (Tensor, optional): Edge weights in case of a weighted graph
                represented as an edge index.
                (default: :obj:`None`).
            l_out (int, optional): Output layer index.
            l_in (int, optional): Input layer index.
            **kwargs: Additional arguments for the spatial message-passing layer.
        """
        raise NotImplementedError

    def stmp(self, x: Tensor, edge_index: Adj,
             edge_weight: Optional[Tensor] = None,
             *,
             layer: int = None,
             **kwargs) -> Tensor:
        temporal_adj = kwargs.pop('temporal_adj', None)
        for l_t in range(self.n_temporal_layers):
            x = self.tmp(x, l_out=layer, l_in=l_t, temporal_adj=temporal_adj)
        for l_s in range(self.n_spatial_layers):
            x = self.smp(x, edge_index, edge_weight, **kwargs,
                         l_out=layer, l_in=l_s)
        return x
