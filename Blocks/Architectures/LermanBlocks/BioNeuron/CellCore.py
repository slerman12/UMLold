import math
import torch
from torch import Tensor
from torch.nn import init, Parameter, Module
from typing import Optional


class Bio(Module):
    neurotransmitters: Optional[Tensor]
    action: Optional[Tensor]
    leak: Optional[Tensor]

    def __init__(self, out_features: int, weight: bool = True,
                 neurotransmitters: bool = False, action: bool = False, leak: bool = False,
                 track_state: bool = False, output_state: bool = False, truncate_interval: int = None,
                 stochastic: bool = False, forward_opt: bool = False) -> None:
        super(Bio, self).__init__()
        assert weight or neurotransmitters
        self.init_weight = weight
        # self.out_features = out_features
        # can also be random 1s/-1s  todo
        for term in ["neurotransmitters", "action", "leak"]:
            if locals()[term]:
                setattr(self, term, Parameter(torch.Tensor(out_features)))
            else:
                self.register_parameter(term, None)

        # (optional) persistence / bptt
        self.track_state = track_state
        self.output_state = output_state
        self.membrane = self.spike = None
        self.truncate_interval = truncate_interval
        self.time_since_truncate = 0
        self.to_truncate = False

        # (optional) sampling and optimization
        self.stochastic = stochastic
        self.forward_opt = forward_opt

    def reset_parameters(self) -> None:
        # todo might be cleaner to move everything weight/bias related to outer module
        if self.init_weight:
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            # can also be random 1s/-1s  todo
            init.constant_(self.weight, 1)
        # init.normal_(self.weight)
        for term in [self.bias, self.action, self.leak]:
            if term is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)  # todo but still need self.weight for this
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(term, -bound, bound)
        # for term in [self.neurotransmitters]:
        #     if term is not None:
        #         init.normal_(term)
        # for term in [self.action, self.leak]:
        #     if term is not None:
        #         init.uniform_(term, -1, 1)
        for term in [self.neurotransmitters]:
            if term is not None:
                init.constant_(term, 1)
        # for term in [self.neurotransmitters]:
        #     if term is not None:
        #         _no_grad_fill_(term[:self.out_features // 2], 1.)
        #         _no_grad_fill_(term[self.out_features // 2:], -1.)
    # def weight_init(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.orthogonal_(m.weight.data)
    #         if hasattr(m.bias, 'data'):
    #             m.bias.data.fill_(0.0)
    #     elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    #         gain = nn.init.calculate_gain('relu')
    #         nn.init.orthogonal_(m.weight.data, gain)
    #         if hasattr(m.bias, 'data'):
    #             m.bias.data.fill_(0.0)

    def reset_state(self) -> None:
        self.membrane = self.spike = None

    # todo can prob get rid of
    def truncate(self):
        self.to_truncate = True

    # todo can prob get rid of ~
    def _truncate(self, membrane=None, spike=None) -> (Tensor, Tensor):
        if membrane is not None:
            membrane = membrane.detach()
        if self.membrane is not None:
            self.membrane = self.membrane.detach()
        if spike is not None:
            spike = spike.detach()
        if self.spike is not None:
            self.spike = self.spike.detach()
        self.time_since_truncate = 0
        self.to_truncate = False
        return membrane, spike

    # todo pass core as input and change this to forward
    def _forward(self, diff: Tensor, prev_membrane: Tensor = None, prev_spike: Tensor = None):
        # todo can prob get rid of
        if self.time_since_truncate == self.truncate_interval or self.to_truncate:
            prev_membrane, prev_spike = self._truncate(prev_membrane, prev_spike)
        if self.truncate_interval is not None:
            self.time_since_truncate += 1

        membrane = 0
        # todo can prob condense
        if prev_membrane is not None:
            membrane = prev_membrane
        elif self.membrane is not None:
            membrane = self.membrane
        if prev_spike is not None:
            membrane = membrane * (1 - prev_spike)
        elif self.spike is not None:
            membrane = membrane * (1 - self.spike)
        if self.leak is not None:
            membrane = membrane * torch.sigmoid(self.leak)
        membrane = membrane + diff

        spike_proba = torch.sigmoid(membrane if self.action is None else membrane + self.action)

        if self.stochastic:
            dist = torch.distributions.bernoulli.Bernoulli(probs=spike_proba)
            spike = spike_proba + dist.sample() - spike_proba.detach()
        else:
            spike = spike_proba + torch.round(spike_proba) - spike_proba.detach()

        output = membrane * spike if self.neurotransmitters is None else self.neurotransmitters * spike

        if self.track_state:
            self.membrane = membrane
            self.spike = spike

        # todo can prob get rid of
        if self.forward_opt:
            # print((spike != 0).any())
            loss = (membrane * spike.detach()).mean()
            # loss = output.mean()  # todo membrane * spike.detach(), or output, or output[spike]?
            loss.backward()
            output = output.detach()

        return output, membrane, spike

