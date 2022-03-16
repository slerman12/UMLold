# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
from torch import nn

from Blocks.Architectures.Residual import Residual

import Utils


"""
A custom "what" pathway: Non-locality stream architectures for BioNet
"""


class Conv2DInvariant(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, bias=True, padding_mode='zeros', num_dilations=4):
        super().__init__(in_channels, out_channels,
                         kernel_size, stride, padding, (1, 1), groups, bias, padding_mode)

        self.num_dilations = num_dilations

    def forward(self, input):
        shape = input.shape
        input = input.view(-1, *shape[-3:])

        convs = []

        for dila in reversed(range(1, self.num_dilations + 1)):
            self.dilation = (dila, dila)  # todo different heights/widths for diff dilations, breaks!

            for _ in range(4):
                convs.append(self._conv_forward(input, self.weight, self.bias))

                # Rotation  # todo check...
                self.weight.data.copy_(torch.rot90(self.weight, dims=[-1, -2]))

        convs = torch.stack(convs, 1)
        return convs.view(*shape[:-3], *convs.shape[1:])


class NonLocalityCNN(nn.Module):
    def __init__(self, input_shape, out_channels=64, groups=8,
                 num_dilations=1, depth=3, output_dim=None):
        super().__init__()

        in_channels = input_shape[0]

        self.trunk = nn.Sequential(
            # Conv2DInvariant(in_channels,
            #                 out_channels, (4, 4), num_dilations=num_dilations),
            nn.Conv2d(in_channels, out_channels, (4, 4)),
            Utils.ChannelSwap(),
            nn.GELU(),
            nn.LayerNorm(out_channels),
            Utils.ChannelSwap()
        )

        self.CNN = nn.Sequential(
            *[Residual(nn.Sequential(
                # Conv2DInvariant(out_channels,
                #                 out_channels, (2, 2), padding='same', groups=groups, num_dilations=num_dilations),
                nn.Conv2d(out_channels, out_channels, (2, 2), padding='same'),
                Utils.ChannelSwap(),
                nn.GELU(),
                nn.LayerNorm(out_channels),
                Utils.ChannelSwap()
            )
            ) for _ in range(depth)])

        # self.CNN = nn.Sequential(
        #     *[Residual(nn.Sequential(
        #         nn.Flatten(0, 1),
        #         nn.Conv2d(out_channels, out_channels, (2, 2), padding='same', groups=groups),
        #         Unflatten(-1, num_dilations * num_rotations, 1, 1, 1),
        #         Utils.ChannelSwap(),
        #         nn.GELU(),
        #         layer_norm(),
        #         Utils.ChannelSwap()
        #     )
        #     ) for _ in range(depth)])

    def feature_shape(self, c, h, w):
        return Utils.cnn_feature_shape(c, h, w, self.trunk)

    def forward(self, x):
        return self.CNN(self.trunk(x))


# class Unflatten(nn.Module):  # Can use einops.layers.torch.Rearrange
#     def __init__(self, *dims):
#         super().__init__()
#         self.dims = dims
#
#     def forward(self, x):
#         return x.view(dim if dim and dim != 1 else x.shape[i] for i, dim in enumerate(self.dims))

# CNNs of various kernal size concatenated channel-wise
# class CrossEmbedLayer(nn.Module):
#     def __init__(
#             self,
#             dim_in,
#             dim_out,
#             kernel_sizes,
#             stride = 2
#     ):
#         super().__init__()
#         kernel_sizes = sorted(kernel_sizes)
#         num_scales = len(kernel_sizes)
#
#         # calculate the dimension at each scale
#         dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
#         dim_scales = [*dim_scales, dim_out - sum(dim_scales)]
#
#         self.convs = nn.ModuleList([])
#         for kernel, dim_scale in zip(kernel_sizes, dim_scales):
#             self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride = stride, padding = (kernel - stride) // 2))
#
#     def forward(self, x):
#         fmaps = tuple(map(lambda conv: conv(x), self.convs))
#         return torch.cat(fmaps, dim = 1)