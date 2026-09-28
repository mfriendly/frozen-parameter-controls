import torch
from torch import nn


class OrthogonalEncoder(nn.Module):
    """Orthogonal encoder layer that projects input features to a higher dimension
    using a semi-orthogonal matrix, such that the input and output vectors have the
    same l2 norm.

    Note: This layer is not trainable and the weights are initialized as a random
    semi-orthogonal matrix."""

    def __init__(self, in_features: int, out_features: int,
                 requires_grad: bool = False):
        super().__init__()
        assert out_features >= in_features
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(out_features, in_features),
                              requires_grad=requires_grad)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the weights as a random orthogonal matrix."""
        nn.init.orthogonal_(self.W)
        assert torch.allclose(self.W.T @ self.W, torch.eye(self.in_features),
                              atol=1e-6), "W is not orthogonal."

    def forward(self, x):
        return x @ self.W.T
