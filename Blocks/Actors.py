# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import copy

from hydra.utils import instantiate

import torch
from torch import nn
from torch.distributions import Categorical

from Distributions import TruncatedNormal

import Utils

from Blocks.Architectures.MLP import MLP


class EnsembleGaussianActor(nn.Module):
    def __init__(self, repr_shape, trunk_dim, hidden_dim, action_dim, recipe, ensemble_size=2,
                 discrete=False, stddev_schedule=None, stddev_clip=None,
                 ema_tau=None, optim_lr=None):
        super().__init__()

        self.discrete = discrete

        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        in_dim = math.prod(repr_shape)  # TODO maybe instead of assuming flattened, should just flatten
        out_dim = action_dim * 2 if stddev_schedule is None else action_dim

        self.trunk = nn.Sequential(nn.Linear(in_dim, trunk_dim),
                                   nn.LayerNorm(trunk_dim), nn.Tanh()) if recipe.trunk._target_ is None \
            else instantiate(recipe.trunk, input_shape=Utils.default(recipe.trunk.input_shape, repr_shape))

        self.Pi_head = Utils.Ensemble([MLP(trunk_dim, out_dim, hidden_dim, 2) if recipe.pi_head._target_ is None
                                       else instantiate(recipe.pi_head, output_dim=out_dim)
                                       for _ in range(ensemble_size)])

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

    def update_ema_params(self):
        assert hasattr(self, 'ema')
        Utils.param_copy(self, self.ema, self.ema_tau)

    def forward(self, obs, step=None):
        obs = self.trunk(obs)

        if self.stddev_schedule is None or step is None:
            mean_tanh, log_stddev = self.Pi_head(obs).chunk(2, dim=-1)
            stddev = torch.exp(log_stddev)
        else:
            mean_tanh = self.Pi_head(obs)
            stddev = torch.full_like(mean_tanh,
                                     Utils.schedule(self.stddev_schedule, step))

        mean = torch.tanh(mean_tanh)
        Pi = TruncatedNormal(mean, stddev, low=-1, high=1, stddev_clip=self.stddev_clip)

        return Pi


class CategoricalCriticActor(nn.Module):  # a.k.a. "Creator"
    def __init__(self, entropy_sched=1):
        super().__init__()

        self.entropy_sched = entropy_sched

    def forward(self, Q, step=None, exploit_temp=1, sample_q=False, actions_log_prob=0):
        # Sample q or mean
        q = Q.rsample() if sample_q else Q.mean

        u = exploit_temp * q + (1 - exploit_temp) * Q.stddev
        u_logits = u - u.max(dim=-1, keepdim=True)[0]
        entropy_temp = Utils.schedule(self.entropy_sched, step)
        Psi = Categorical(logits=u_logits / entropy_temp + actions_log_prob)

        best_eps, best_ind = torch.max(u, -1)
        best_action = Utils.gather_indices(Q.action, best_ind.unsqueeze(-1), 1).squeeze(1)

        sample = Psi.sample

        def action_sampler(sample_shape=torch.Size()):
            i = sample(sample_shape)
            return Utils.gather_indices(Q.action, i.unsqueeze(-1), 1).squeeze(1)

        Psi.__dict__.update({'best': best_action,
                             'best_u': best_eps,
                             'sample_ind': sample,
                             'sample': action_sampler,
                             'Q': Q,
                             'q': q,
                             'actions': Q.action,
                             'u': u})
        return Psi
