import torch
from torch import nn
import numpy as np

import normflows as nf
from normflows.flows.base import Flow

from .MaskedPiecewiseRationalQuadraticAutoregressive import MaskedPiecewiseRationalQuadraticAutoregressive

class AutoregressiveRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer, wrapper for the implementation
    of Durkan et al., see [sources](https://github.com/bayesiains/nsf)
    """

    def __init__(
        self,
        num_input_channels,
        num_blocks,
        num_hidden_channels,
        num_context_channels=None,
        num_bins=8,
        tail_bound=3,
        activation=nn.ReLU,
        dropout_probability=0.0,
        permute_mask=False,
        init_identity=False,
    ):
        """Constructor

        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          num_context_channels (int): Number of context/conditional channels
          num_bins (int): Number of bins
          tail_bound (int): Bound of the spline tails
          activation (torch.nn.Module): Activation function
          dropout_probability (float): Dropout probability of the NN
          permute_mask (bool): Flag, permutes the mask of the NN
          init_identity (bool): Flag, initialize transform as identity
        """
        super().__init__()

        self.mprqat = MaskedPiecewiseRationalQuadraticAutoregressive(
            features=num_input_channels,
            hidden_features=num_hidden_channels,
            context_features=num_context_channels,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            num_blocks=num_blocks,
            use_residual_blocks=True,
            random_mask=False,
            permute_mask=permute_mask,
            activation=activation(),
            dropout_probability=dropout_probability,
            use_batch_norm=False,
            init_identity=init_identity,
        )

    def forward(self, z, context=None):
        z, log_det = self.mprqat.inverse(z, context=context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, log_det = self.mprqat(z, context=context)
        return z, log_det.view(-1)