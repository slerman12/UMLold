import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from .. import Bio


# todo can prob just extend Linear, pass in bio cell, call super as needed (e.g. str rep)
class BioLinear(Bio):
    r"""Applies a bio-inspired spiking transmit from the incoming data

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        weight: If set to ``False``, the layer will not learn its weights, then must set neurotransmitters to True
            Default: ``True``
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        neurotransmitters: If set to ``False``, the layer will not learn quantized neurotr.
            Default: ``False``
        action: If set to ``False``, the layer will not learn an action potential.
            Default: ``False``
        leak: If set to ``False``, the layer will not learn a leak factor.
            Default: ``False``
        track_state: If set to ``True``, the layer will handle membrane/spike state storage/reuse automatically.
            Default: ``False``
        output_state: If set to ``False``, the layer will not output the membrane/spike state.
            Default: ``False``
        truncate_interval:  if not set to ``None``, will periodically "truncate" (detach) the sequence backprop
            Default: ``None``
        stochastic:  if set to ``True``, will spike stochastically
            Default: ``False``
        forward_opt: if set to ``True'', will compute forward-pass "gradients" for forwardprop

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`.
          Additionally can accept ``prev_membrane`` and ``prev_spike`` each of shape :math:`(N, *, H_{out})`
        - Output: 3-element tuple with each element of :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
          Outputs correspond to ``(y, membrane, spike``)``.
          Or, if ``output_state`` set to ``False``, outputs just ``y``.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
        neurotransmitters:   the learnable neurotransmitters of the module of shape :math:`(\text{out\_features})`.
                initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
        action:   the learnable action potential of the module of shape :math:`(\text{out\_features})`.
                If :attr:`action` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
        leak:   the learnable leak factor of the module of shape :math:`(\text{out\_features})`.
                If :attr:`leak` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = BioLinear(20, 30)
        >>> x = torch.randn(128, 20)
        >>> y = m(x)
        >>> print(y.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, weight: bool = True, bias: bool = True,
                 neurotransmitters: bool = False, action: bool = False, leak: bool = False,
                 track_state: bool = False, output_state: bool = False, truncate_interval: int = None,
                 stochastic: bool = False, forward_opt: bool = False) -> None:
        super(BioLinear, self).__init__(out_features, weight, neurotransmitters, action, leak,
                                        track_state, output_state, truncate_interval, stochastic, forward_opt)
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.Tensor(out_features, in_features)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        if weight:
            self.weight = Parameter(self.weight)

    def forward(self, input: Tensor, prev_membrane: Tensor = None, prev_spike: Tensor = None):
        # diff = F.linear(x, torch.sigmoid(self.weight))  # todo
        # diff = F.linear(x, torch.ones_like(self.weight))  # todo
        diff = F.linear(input, self.weight, self.bias)

        y, membrane, spike = self._forward(diff, prev_membrane, prev_spike)

        if self.output_state:
            return y, membrane, spike
        else:
            return y

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, action={}, leak={}, track_state={}, output_state={}, ' \
               'truncate_interval={}'.format(self.in_features, self.out_features, self.bias is not None,
                                             self.action is not None, self.leak is not None,
                                             self.track_state, self.output_state, self.truncate_interval
                                             )

