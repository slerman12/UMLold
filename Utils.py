# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import random
import re
import warnings
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# Sets all Pytorch and Numpy random seeds
def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# Saves module + attributes
def save(path, module, **attributes):
    path = path.replace('Agents.', '')
    Path('/'.join(path.split('/')[:-1])).mkdir(exist_ok=True, parents=True)
    attributes.update({'state_dict': module.state_dict()})
    torch.save(attributes, path)


# Loads module
def load(path, module):
    fetch = True
    while fetch:
        try:
            path = path.replace('Agents.', '')
            if Path(path).exists():
                to_load = torch.load(path, map_location=module.device)
                module.load_state_dict(to_load['state_dict'], strict=False)
                del to_load['state_dict']
                for key in to_load:
                    setattr(module, key, to_load[key])
            else:
                warnings.warn(f'Load path {path} does not exist.')
            fetch = False
        except:
            warnings.warn(f'Load conflict')  # For distributed training
            pass


# Assigns a default value to x if x is None
def default(x, value):
    if x is None:
        x = value
    return x


# Initializes model weights according to common distributions
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


# Copies parameters from one model to another, with optionally EMA weighing
def param_copy(net, target_net, ema_tau=1):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(ema_tau * param.data +
                                (1 - ema_tau) * target_param.data)


# Compute the output shape of a CNN layer
def cnn_layer_feature_shape(in_height, in_width, kernel_size=1, stride=1, padding=0, dilation=1):
    if padding == 'same':
        return in_height, in_width
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) is not tuple:
        stride = (stride, stride)
    if type(padding) is not tuple:
        padding = (padding, padding)
    if type(dilation) is not tuple:
        dilation = (dilation, dilation)
    out_height = math.floor(((in_height + (2 * padding[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1) / stride[0]) + 1)
    out_width = math.floor(((in_width + (2 * padding[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1) / stride[1]) + 1)
    return out_height, out_width


# Compute the output shape of a whole CNN
def cnn_feature_shape(channels, height, width, *blocks):
    for block in blocks:
        if isinstance(block, (nn.Conv2d, nn.AvgPool2d)):
            channels = block.out_channels
            height, width = cnn_layer_feature_shape(height, width,
                                                    kernel_size=block.kernel_size,
                                                    stride=block.stride,
                                                    padding=block.padding)
        elif isinstance(block, nn.Linear):
            channels = block.out_features  # Assumes channels-last if linear
        elif isinstance(block, nn.Flatten) and block.start_dim == -3:
            channels, height, width = channels * height * width, 1, 1  # Placeholder height/width dims
        elif isinstance(block, nn.AdaptiveAvgPool2d):
            height, width = block.output_size
        elif hasattr(block, 'repr_shape'):
            channels, height, width = block.repr_shape(channels, height, width)
        elif hasattr(block, 'modules'):
            for module in block.children():
                channels, height, width = cnn_feature_shape(channels, height, width, module)

    feature_shape = (channels, height, width)  # TODO should probably do (channels, width, height) universally

    return feature_shape


# Broadcasts a CNN's inputs and channel-wise-context to a desired input shape, preserving leading dims
class CNNInputBroadcast(nn.Module):
    def __init__(self, module, desired_input_shape):
        super().__init__()

        self.module = module
        self.shape = desired_input_shape

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        x = torch.cat(
            [context.view(*context.shape[:-3], -1, *self.shape[1:]) if len(context.shape) > 3
             else context.view(*context.shape[:-1], -1, *self.shape[1:]) if context.shape[-1]
                                                                            % math.prod(self.shape[1:]) == 0
             else context.view(*context.shape, 1, 1).expand(*context.shape, *self.shape[1:])
             for context in x if context.nelement() > 0], dim=-3)
        # Conserve leading dims
        lead_shape = x.shape[:-3]
        # Operate on last 3 dims
        x = x.view(-1, *x.shape[-3:])

        x = self.module(x)

        # Restore leading dims
        out = x.view(*lead_shape, *x.shape[1:])
        return out


# "Ensembles" (stacks) multiple modules' outputs
class Ensemble(nn.Module):
    def __init__(self, modules, dim=1):
        super().__init__()

        self.ensemble = nn.ModuleList(modules)
        self.dim = dim

    def forward(self, *x):
        return torch.stack([module(*x) for module in self.ensemble],
                           self.dim)


# Merges multiple critics into one if so desired (ensembles of ensembles)
class MergeCritics(nn.Module):
    def __init__(self, *critics):
        super().__init__()
        self.critics = critics

    def forward(self, obs, action=None, context=None):
        Q = [critic(obs, action, context) for critic in self.critics]
        Qs = torch.cat([Q_.Qs for Q_ in Q], 0)
        # Dist
        stddev, mean = torch.std_mean(Qs, dim=0)
        merged_Q = Normal(mean, stddev + 1e-12)
        merged_Q.__dict__.update({'Qs': Qs,
                                  'action': Q[0].action})
        return merged_Q


# Replaces tensor's batch items with Normal-sampled random latent
class Randn(nn.Module):
    def __init__(self, size=1):
        super().__init__()
        self.size = size

    def forward(self, x):
        return torch.randn((x.shape[0], self.size), device=x.device)


# Replaces tensor's batch items with Uniform-sampled random latent
class Rand(Randn):
    def forward(self, x):
        return torch.rand((x.shape[0], self.size), device=x.device)


# (Multi-dim) one-hot encoding
def one_hot(x, num_classes):
    # assert x.shape[-1] == 1
    x = x.squeeze(-1).unsqueeze(-1)  # Or this
    x = x.long()
    shape = x.shape[:-1]
    zeros = torch.zeros(*shape, num_classes, dtype=x.dtype, device=x.device)
    return zeros.scatter(len(shape), x, 1).float()


# Differentiable one_hot
def rone_hot(x):
    return x - (x - one_hot(torch.argmax(x, -1, keepdim=True), x.shape[-1]))


# Differentiable clamp
def rclamp(x, min, max):
    return x - (x - torch.clamp(x, min, max))


# (Multi-dim) indexing
def gather_indices(item, ind, dim=-1):
    ind = ind.long().expand(*item.shape[:dim], ind.shape[-1])  # Assumes ind.shape[-1] is desired num indices
    if -1 < dim < len(item.shape) - 1:
        trail_shape = item.shape[dim + 1:]
        ind = ind.reshape(ind.shape + (1,)*len(trail_shape))
        ind = ind.expand(*ind.shape[:dim + 1], *trail_shape)
    return torch.gather(item, dim, ind)


# Basic L2 normalization
class L2Norm(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, dim=-1, eps=self.eps)


# Min-max normalizes to [0, 1]
# "Re-normalization", as in SPR (https://arxiv.org/abs/2007.05929), or at least in their code
class ShiftMaxNorm(nn.Module):
    def __init__(self, start_dim=-1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x):
        y = x.flatten(self.start_dim)
        y = y - y.min(-1, keepdim=True)[0]
        y = y / y.max(-1, keepdim=True)[0]
        return y.view(*x.shape)


# Swaps image dims between channel-last and channel-first format,
# or does this conservatively before/after running sequence of modules
class ChannelSwap(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.Ms = nn.Sequential(*modules)

    def forward(self, x):
        if len(self.Ms):
            x = self.Ms(x.transpose(-1, -3))
        return x.transpose(-1, -3)


# Context manager that temporarily switches on eval() mode for specified models; then resets them
class act_mode:
    def __init__(self, *models):
        super().__init__()
        self.models = models

    def __enter__(self):
        self.start_modes = []
        for model in self.models:
            if model is None:
                self.start_modes.append(None)
            else:
                self.start_modes.append(model.training)
                model.eval()

    def __exit__(self, *args):
        for model, mode in zip(self.models, self.start_modes):
            if model is not None:
                model.train(mode)
        return False


# Converts data to Torch Tensors and moves them to the specified device as floats
def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device).float() for x in xs)


# Backward pass on a loss; clear the grads of models; update EMAs; step optimizers
def optimize(loss=None, *models, clear_grads=True, backward=True, retain_graph=False, step_optim=True, ema=True):
    # Clear grads
    if clear_grads and loss is not None:
        for model in models:
            model.optim.zero_grad(set_to_none=True)

    # Backward
    if backward and loss is not None:
        loss.backward(retain_graph=retain_graph)

    # Optimize
    if step_optim:
        for model in models:
            model.optim.step()

            # Update ema target
            if ema and hasattr(model, 'ema'):
                model.update_ema_params()

            if loss is None and clear_grads:
                model.optim.zero_grad(set_to_none=True)


# Increment/decrement a value in proportion to a step count based on a string-formatted schedule (only supports linear)
def schedule(schedule, step):
    try:
        return float(schedule)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schedule)
        if match:
            start, stop, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * start + mix * stop


