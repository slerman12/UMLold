import math

import torch
from torch import nn

import Utils


class CNN(nn.Module):
    def __init__(self, input_shape, out_channels=32, depth=3, batch_norm=False, output_dim=None):
        super().__init__()

        self.input_shape = torch.Size(input_shape)
        in_channels = input_shape[0]

        self.trunk = nn.Identity()

        self.CNN = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(in_channels if i == 0 else out_channels,
                                      out_channels, 3, stride=2 if i == 0 else 1),
                            nn.BatchNorm2d(self.out_channels) if batch_norm else nn.Identity(),
                            nn.ReLU()) for i in range(depth + 1)],
        )

        self.projection = nn.Identity() if output_dim is None \
            else nn.Sequential(nn.AdaptiveAvgPool2d((3, 3)),
                               nn.Flatten(-3),
                               nn.Linear(out_channels * 3 * 3, 1024),
                               nn.ReLU(inplace=True),
                               nn.Linear(1024, output_dim))

        self.apply(Utils.weight_init)

    def repr_shape(self, c, h, w):
        return Utils.cnn_feature_shape(c, h, w, self.trunk, self.CNN, self.projection)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        x = torch.cat(
            [context.view(*context.shape[:-3], -1, *self.input_shape[1:]) if len(context.shape) > 3
             else context.view(*context.shape[:-1], -1, *self.input_shape[1:]) if context.shape[-1]
                                                                                  % math.prod(self.input_shape[1:]) == 0
             else context.view(*context.shape, 1, 1).expand(*context.shape, *self.input_shape[1:])
             for context in x if context.nelement() > 0], dim=-3)
        # Conserve leading dims
        lead_shape = x.shape[:-3]
        # Operate on last 3 dims
        x = x.view(-1, *x.shape[-3:])

        x = self.trunk(x)
        x = self.CNN(x)
        x = self.projection(x)

        # Restore leading dims
        out = x.view(*lead_shape, *x.shape[1:])
        return out


# class SimpleDecoder(nn.Module):
#     def __init__(self, out_shape, depth=3):
#         super().__init__()
#
#         channels_in = 3
#         channels_out = out_shape[0]
#
#         self.CNN = nn.Sequential(*[nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(channels_in // 2 ** (i - 1), channels_out if i == depth else channels_in // 2 ** i, 3, 1, 1),
#             nn.GLU(1)) for i in range(depth + 1)])
#
#     def forward(self, x):
#         return self.CNN(x)
