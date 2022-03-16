# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn
from opt_einsum_torch import EinsumPlanner
import copy
from einops import rearrange

from Blocks.Architectures.MLP import MLP

import Utils


class CrossAttention(nn.Module):
    def __init__(self, dim=32, heads=None, context_dim=None, value_dim=None, ln_v=False, talk_h=False):
        super().__init__()

        self.dim = dim

        context_dim = dim if context_dim is None \
            else context_dim

        value_dim = dim if value_dim is None \
            else value_dim

        heads = math.gcd(8, value_dim) if heads is None \
            else heads

        self.value_dim = value_dim
        self.heads = heads

        assert value_dim % heads == 0, f'value dim={dim} is not divisible by heads={heads}'

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim + value_dim, bias=False)

        self.ln_v = nn.LayerNorm(value_dim // heads) if ln_v \
            else nn.Identity()

        # "Talking heads" (https://arxiv.org/abs/2003.02436)
        self.talk_h = nn.Sequential(Utils.ChSwap, nn.Linear(heads, heads, bias=False),
                                    nn.LayerNorm(heads), Utils.ChSwap) if talk_h else nn.Identity()

    def forward(self, x, context=None):
        # Conserves shape
        shape = x.shape
        assert shape[-1] == self.dim, f'input dim ≠ pre-specified {shape[-1]}≠{self.dim}'

        if context is None:
            context = x

        x = x.flatten(1, -2)
        context = context.flatten(1, -2)

        q = self.to_q(x)
        k, v = self.to_kv(context).tensor_split([self.dim], dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        v = self.ln_v(v)  # My variant, makes sure the values get an equal "vote"

        # Memory efficient toggle, e.g., =0.5
        mem_limit = False
        einsum = EinsumPlanner(q.device, cuda_mem_limit=mem_limit).einsum if 0 < mem_limit < 1 \
            else torch.einsum

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.dim ** -0.5

        attn = dots.softmax(dim=-1)

        # "Talking heads"
        attn = self.talk_h(attn)

        attn = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(attn, 'b h n d -> b n (h d)')

        # Restores original shape
        out = out.view(shape)

        if 0 < mem_limit < 1:
            out = out.to(q.device)

        return out


# A minimalist implementation using only Pytorch natives
# class CrossAttention(nn.Module):
#     def __init__(self, dim=32, heads=8, context_dim=None, *_):
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
    def forward(self, x, *_):
        return super().forward(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=32, heads=1, context_dim=None, value_dim=None, ln_input=False, ln_v=True, talk_h=False,
                 optim_lr=None, ema_tau=None):
        super().__init__()

        context_dim = dim if context_dim is None \
            else context_dim

        value_dim = dim if value_dim is None \
            else value_dim

        self.heads = math.gcd(8, value_dim) if heads is None \
            else heads

        self.value_dim = value_dim

        self.attn = CrossAttention(dim, self.heads, context_dim, value_dim,
                                   ln_v,  # My variant, norms value heads by default
                                   talk_h)
        self.mlp = MLP(value_dim, value_dim, value_dim, 1, nn.GELU())

        self.ln_input = nn.LayerNorm(dim) if ln_input else nn.Identity()  # My variant, but default False
        self.ln_attn = nn.LayerNorm(context_dim)
        self.ln = nn.LayerNorm(value_dim)

        self.init(optim_lr, ema_tau)

    def repr_shape(self, c, h, w):
        return self.value_dim, h, w  # Assumes channels last

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

    def forward(self, x, context=None):
        # if context is None:
        #     context = x
        #
        # attn = self.ln_attn(self.attn(x, context)) + x
        # out = self.ln(self.mlp(attn)) + attn
        #
        # return out
        """My variant"""
        x = self.ln_input(x)  # TODO This might help too

        if context is None:
            context = x

        # A key idea here is the layer-norm IN the attention, ln(values) rather than ln(attn)
        attn = self.attn(x, context)  # If residual here, then attn="candidate update" with fc="correction" to x
        fc = self.mlp(attn)  # This way MLP can help relationally-reason rather than just non-locally course-correct
        ln = self.ln(fc)
        return ln + x  # This variant can do relational reasoning, more capacity; THEN does a residual-based update


class SelfAttentionBlock(CrossAttentionBlock):
    def forward(self, x, *_):
        return super().forward(x)


class AttentionPool(nn.Module):
    def __init__(self, channels_in=32, heads=None, output_dim=None, depth=1, recursions=0, input_shape=None):
        super().__init__()

        self.input_shape = input_shape

        if input_shape is not None:
            channels_in = input_shape[-3]

        if output_dim is None:
            output_dim = channels_in

        if heads is None:
            heads = math.gcd(output_dim, 8)  # Approx 8

        self.pool = nn.Sequential(Utils.ChSwap,
                                  # Alternatively could also recurse
                                  *([SelfAttentionBlock(channels_in, heads)] * recursions),
                                  # "Transformer"
                                  *[SelfAttentionBlock(dim=channels_in if i == 0 else output_dim, heads=heads,
                                                       value_dim=output_dim) for i in range(depth)],
                                  Utils.ChSwap,
                                  nn.AdaptiveAvgPool2d((1, 1)),
                                  nn.Flatten(-3))

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
