# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import copy
import math

import torch
from hydra.utils import instantiate
from torch import nn

import Utils
from Blocks.Architectures.Vision.CNN import CNN
from Blocks.Architectures.Vision.ResNet import MiniResNet


class CNNEncoder(nn.Module):
    """
    Basic CNN encoder, e.g., DrQV2 (https://arxiv.org/abs/2107.09645).
    """

    def __init__(self, obs_shape, out_channels=32, depth=3, batch_norm=False, shift_max_norm=False, pixels=True,
                 recipe=None, optim_lr=None, ema_tau=None, parallel=False):

        super().__init__()

        assert len(obs_shape) == 3, 'image observation shape must have 3 dimensions'

        self.obs_shape = obs_shape
        self.pixels = pixels

        # CNN
        self.Eyes = nn.Sequential(CNN(obs_shape, out_channels, depth, batch_norm) if recipe.eyes._target_ is None
                                  else instantiate(recipe.eyes),
                                  Utils.ShiftMaxNorm(-3) if shift_max_norm else nn.Identity())
        if parallel:
            self.Eyes = nn.DataParallel(self.Eyes)  # Parallel on visible GPUs

        self.pool = nn.Flatten(-3) if recipe.pool._target_ is None \
            else instantiate(recipe.pool, input_shape=self._feature_shape())

        # Initialize model
        self.init(optim_lr, ema_tau)

    def _feature_shape(self):
        return Utils.cnn_feature_shape(*self.obs_shape, self.Eyes)

    def init(self, optim_lr=None, ema_tau=None):
        # Initialize weights
        self.apply(Utils.weight_init)

        # Optimizer
        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        # Dimensions
        self.feature_shape = self._feature_shape()  # Feature map shape

        self.repr_shape = Utils.cnn_feature_shape(*self.feature_shape, self.pool)
        self.repr_dim = math.prod(self.repr_shape)  # Flattened repr dim

        # EMA
        if ema_tau is not None:
            self.ema = copy.deepcopy(self)
            self.ema_tau = ema_tau

    def update_ema_params(self):
        assert hasattr(self, 'ema')
        Utils.param_copy(self, self.ema, self.ema_tau)

    # Encodes
    def forward(self, obs, *context, flatten=True):
        obs_shape = obs.shape  # Preserve leading dims
        assert obs_shape[-3:] == self.obs_shape, f'encoder received an invalid obs shape {obs_shape}'
        obs = obs.flatten(0, -4)  # Encode last 3 dims

        # Normalizes pixels
        if self.pixels:
            obs = obs / 127.5 - 1

        # Optionally append context to channels assuming dimensions allow
        context = [c.reshape(obs.shape[0], c.shape[-1], 1, 1).expand(-1, -1, *self.obs_shape[1:])
                   for c in context]
        obs = torch.cat([obs, *context], 1)

        # CNN encode
        h = self.Eyes(obs)

        # Restore leading dims
        h = h.view(*obs_shape[:-3], *h.shape[-3:])
        assert tuple(h.shape[-3:]) == self.feature_shape, f'pre-computed feature_shape does not match output CNN shape' \
                                                          f'{self.feature_shape}≠{tuple(h.shape[-3:])}'

        if flatten:
            h = self.pool(h)
            assert h.shape[-1] == self.repr_dim or tuple(h.shape[-3:]) == self.repr_shape, \
                f'pre-computed repr_dim/repr_shape does not match output dim ' \
                f'{self.repr_dim}≠{h.shape[-1]}, {self.repr_shape}≠{tuple(h.shape[-3:])}'
        return h


class ResidualBlockEncoder(CNNEncoder):
    """
    Residual block-based CNN encoder,
    Isotropic means no bottleneck / dimensionality conserving
    """

    def __init__(self, obs_shape, context_dim=0, out_channels=32, hidden_channels=64, num_blocks=1, shift_max_norm=True,
                 pixels=True, isotropic=False, recipe=None, optim_lr=None, ema_tau=None):

        super().__init__(obs_shape, hidden_channels, 0, pixels)

        # Dimensions
        in_channels = obs_shape[0] + context_dim
        out_channels = obs_shape[0] if isotropic else out_channels

        # CNN ResNet-ish
        self.Eyes = nn.DataParallel(  # ! Automatically parallel across visible GPUs
            nn.Sequential(MiniResNet((in_channels, *obs_shape[1:]), 2 - isotropic,
                                     [hidden_channels, out_channels], [num_blocks]),
                          Utils.ShiftMaxNorm(-3) if shift_max_norm else nn.Identity()))

        self.init(optim_lr, ema_tau)

        # Isotropic
        if isotropic:
            assert tuple(obs_shape) == self.feature_shape, \
                f'specified to be isotropic, but in {tuple(obs_shape)} ≠ out {self.feature_shape}'
