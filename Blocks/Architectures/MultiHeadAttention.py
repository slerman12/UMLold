# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn
from torch import einsum
from opt_einsum_torch import EinsumPlanner
import copy
from einops import rearrange

from Blocks.Architectures.MLP import MLP

import Utils


class CrossAttention(nn.Module):
    def __init__(self, dim=32, heads=8, context_dim=None, talk_h=False):
        super().__init__()

        assert dim % heads == 0, f'dim={dim} does not divide heads={heads}'
        self.dim = dim
        self.heads = heads

        context_dim = dim if context_dim is None \
            else context_dim

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim * 2, bias=False)

        # "Talking heads" (https://arxiv.org/abs/2003.02436)
        self.talk_h = nn.Sequential(Utils.ChannelSwap(),
                                    nn.Linear(heads, heads, bias=False),
                                    nn.LayerNorm(heads), Utils.ChannelSwap()) if talk_h else nn.Identity()

    def forward(self, x, context):
        # Conserves shape
        shape = x.shape
        assert shape[-1] == self.dim, f'{shape[-1]}, {self.dim}'

        x = x.flatten(1, -2)
        context = context.flatten(1, -2)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        # device = torch.device(q.device)

        # ee = EinsumPlanner(device, cuda_mem_limit=0.2)
        # dots = ee.einsum('b h i d, b h j d -> b h i j', q, k) * self.dim ** -0.5

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.dim ** -0.5

        attn = dots.softmax(dim=-1)

        # "Talking heads"
        attn = self.talk_h(attn)

        # ee = EinsumPlanner(device, cuda_mem_limit=0.2)
        # out = ee.einsum('b h i j, b h j d -> b h i d', attn, v) * self.dim ** -0.5

        attn = einsum('b h i j, b h j d -> b h i d', attn, v)  # todo maybe nn.MultiHeadAttention?

        out = rearrange(attn, 'b h n d -> b n (h d)')

        # Restores original shape
        out = out.view(shape)
        # return out.view(shape).to(device)

        return out


# A minimalist implementation using only Pytorch natives
# class CrossAttention(nn.Module):
#     def __init__(self, dim=32, heads=8, context_dim=None, talk_h=False):
#         super().__init__()
#
#         assert dim % heads == 0, f'dim={dim} does not divide heads={heads}'
#         self.dim = dim
#
#         self.attn = nn.MultiheadAttention(dim, heads, kdim=context_dim, vdim=context_dim, batch_first=True)
#
#     def forward(self, x, context):
#         # Conserves shape
#         shape = x.shape
#         assert shape[-1] == self.dim, f'{shape[-1]}, {self.dim}'
#
#         x = x.flatten(1, -2)
#         context = context.flatten(1, -2)
#
#         attn, _ = self.attn(x, context, context)
#
#         # Restores original shape
#         return attn.view(shape)


class SelfAttention(CrossAttention):
    def forward(self, x, *args):
        return super().forward(x, x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=32, heads=8, context_dim=None, talk_h=False, optim_lr=None, ema_tau=None):
        super().__init__()

        self.dim = dim
        self.heads = heads

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, heads, context_dim, talk_h)
        self.mlp = MLP(dim, dim, dim, 2, nn.GELU())

        self.init(optim_lr, ema_tau)

    def init(self, optim_lr=None, ema_tau=None):
        # Initialize weights
        self.apply(Utils.weight_init)

        # Optimizer
        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        # EMA
        if ema_tau is not None:
            self.ema = copy.deepcopy(self)
            self.ema_tau = ema_tau

    def forward(self, x, context):
        attn = self.ln1(self.attn(x, context)) + x
        out = self.ln2(self.mlp(attn)) + attn

        return out


class SelfAttentionBlock(CrossAttentionBlock):
    def forward(self, x, *_):
        return super().forward(x, x)


class AttentionPool(nn.Module):
    def __init__(self, channels_in=32, heads=None, output_dim=None, input_shape=None):
        super().__init__()

        self.channels_in = channels_in
        self.input_shape = input_shape

        if input_shape is not None:
            channels_in = input_shape[-3] if len(input_shape) >= 3 else input_shape[-1]

        if heads is None:
            heads = (channels_in + 4) // 4

        self.pool = nn.Sequential(Utils.ChannelSwap(),
                                  SelfAttentionBlock(dim=channels_in, heads=heads),
                                  Utils.ChannelSwap(),
                                  nn.AdaptiveAvgPool2d((1, 1)),
                                  nn.Flatten(),
                                  nn.Linear(channels_in, channels_in if output_dim is None else output_dim))

    def repr_shape(self, c, h, w):
        return Utils.cnn_feature_shape(c, h, w, self.pool)

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

        x = self.pool(x)

        # Restore leading dims
        out = x.view(*lead_shape, *x.shape[1:])
        return out
