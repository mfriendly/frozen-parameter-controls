from torch import nn
from tsl import logger
from tsl.nn import get_functional_activation


class MPStack(nn.Module):
    def __init__(self,
                 gnn_class: nn.Module,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int = 1,
                 activation: str = None,
                 **gnn_kwargs):
        super().__init__()
        self.gnn_class = gnn_class
        self.gnn_kwargs = gnn_kwargs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.activation = get_functional_activation(activation)

        self.gnns = nn.ModuleList([
            gnn_class(input_size if i == 0 else hidden_size,  # in_channels
                      hidden_size,  # out_channels
                      **gnn_kwargs)
            for i in range(n_layers)
        ])

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        for layer in self.gnns:
            x = layer(x, edge_index, edge_weight, **kwargs)
            x = self.activation(x)
        return x

    def reset_parameters(self):
        for layer in self.gnns:
            layer.reset_parameters()


class CheckReceptiveField(nn.Module):
    def __init__(self, receptive_field: int, dim: int = 1):
        super().__init__()
        self.receptive_field = receptive_field
        self.dim = dim
        self._checked_lengths = set()

    def forward(self, x):
        length = x.size(self.dim)
        if length <= self.receptive_field:
            return x
        # Cut input to receptive field length
        x = x[:, -self.receptive_field:]
        if length not in self._checked_lengths:
            self._checked_lengths.add(length)
            logger.warning(f"Input is {length}-steps long, but receptive field "
                           f"is {self.receptive_field}!")
        return x
