import math
import torch
from torch import Tensor
from torch.nn import Parameter, init
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import SGD


class ProbasLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, lr: float = 0.01) -> None:
        super(ProbasLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = self.bias = None

        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = Parameter(torch.full((out_features, in_features), lr))
        self.weight_m = Parameter(torch.Tensor(out_features, in_features))
        self.weight_b = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_sigma = Parameter(torch.full((out_features,), lr))
            self.bias_m = Parameter(torch.Tensor(out_features))
            self.bias_b = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.opt = SGD(self.parameters(), lr=lr)

    def reset_parameters(self) -> None:
        for w in [self.weight_mu, self.weight_m, self.weight_b]:
            init.kaiming_uniform_(w, a=math.sqrt(5))
        if self.bias is not None:
            for b in [self.bias_mu, self.bias_m, self.bias_b]:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(b, -bound, bound)

    def weight_dist(self):
        return Normal(self.weight_mu, torch.abs(self.weight_sigma))

    def bias_dist(self):
        return Normal(self.bias_mu, torch.abs(self.bias_sigma)) if self.bias is not None else None

    # def _loss(self, utility):
    #     assert self.weight is not None, "model optimizer requires at least one forward pass with model in training mode"
    #     utility_probas = torch.softmax(utility, dim=0)
    #
    #     weight_probas = torch.softmax(self.weight_dist().log_prob(self.weight), dim=0)
    #     # weight_probas = weight_probas.mean(2).mean(1)
    #     loss = F.kl_div(weight_probas, utility_probas[:, None, None].expand_as(weight_probas))
    #
    #     if self.bias is not None:
    #         bias_probas = torch.softmax(self.bias_dist().log_prob(self.bias), dim=0)
    #         # bias_probas = bias_probas.mean(1)
    #         loss = loss + F.kl_div(bias_probas, utility_probas[:, None].expand_as(bias_probas))
    #
    #     return loss
    #
    # def _loss(self, utility):
    #     assert self.weight is not None, "model optimizer requires at least one forward pass with model in training mode"
    #     loss = F.mse_loss(self.weight_m[None, :, :] * self.weight + self.weight_b[None, :, :],
    #                       utility[:, None, None].expand_as(self.weight))
    #
    #     if self.bias is not None:
    #         loss = loss + F.mse_loss(self.bias_m[None, :] * self.bias + self.bias_b[None, :],
    #                                  utility[:, None].expand_as(self.bias))
    #
    #     return loss

    def _loss(self, utility):
        assert self.weight is not None, "model optimizer requires at least one forward pass with model in training mode"
        weight_probas = torch.softmax(self.weight_dist().log_prob(self.weight), dim=0)
        loss = -(weight_probas * utility[:, None, None].expand_as(self.weight)).mean()

        if self.bias is not None:
            pass

        return loss

    def forward(self, input: Tensor, utility=None):
        if utility is not None:
            self.opt.zero_grad()
            self._loss(utility.detach()).backward()
            self.opt.step()
        if input is not None:
            if self.training:
                input_ = input.view(-1, input.shape[-1])

                self.weight = self.weight_dist().rsample(sample_shape=[input_.shape[0]])
                output = torch.einsum('bi,bji->bj', input_, self.weight)

                if self.bias is not None:
                    self.bias = self.bias_dist().rsample(sample_shape=[input_.shape[0]])
                    output = output + self.bias
                return output.view(*input.shape[:-1], -1)
            else:
                return F.linear(input, self.weight_mu, self.bias_mu)

