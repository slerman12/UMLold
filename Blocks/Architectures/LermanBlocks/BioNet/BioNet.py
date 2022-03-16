# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from torch import nn

import Utils

from Blocks.Architectures import ResNet, MLP
from Blocks.Architectures.MultiHeadAttention import CrossAttentionBlock


class BioNetV1(nn.Module):
    """Disentangling "What" And "Where" Pathways In CNNs
        - V1: Uses two ResNets"""

    def __init__(self, input_shape, out_channels=32, heads=8, output_dim=None):
        super().__init__()
        resnet_dims, resnet_depths = [64, 64, 128, 256, 512], [2, 2, 2, 2]

        self.ventral_stream = ResNet(input_shape, 3, 2, dims=resnet_dims, depths=resnet_depths)
        self.dorsal_stream = ResNet(input_shape, 3, 2, dims=resnet_dims, depths=resnet_depths)

        self.cross_talk = nn.ModuleList([CrossAttentionBlock(dim=dim, heads=heads)
                                         for dim in resnet_dims[1:]])

        self.projection = nn.Identity() if output_dim is None \
            else nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                               nn.Flatten(),
                               MLP(out_channels, output_dim, 1024))

    def repr_shape(self, c, h, w):
        return Utils.cnn_feature_shape(c, h, w, self.dorsal_stream, self.projection)

    def forward(self, input):
        ventral = self.ventral_stream.trunk(input)
        dorsal = self.dorsal_stream.trunk(input)

        t = Utils.ChannelSwap()  # Swaps between channels-first channels-last format

        for what, where, talk in zip(self.ventral_stream.ResNet,
                                     self.dorsal_stream.ResNet,
                                     self.cross_talk):
            ventral = what(ventral)
            dorsal = t(talk(t(where(dorsal)),
                            t(ventral)))

        out = self.projection(dorsal)
        return out

    """Aside from the way the modules are put together (via two disentangled streams), 
    there is not much that we have not seen before in some form
    
    We have a CNN [resnet, convmixer], ViT [worth 1000 words], CrossAttention [perceiver], 
    SelfAttention [all you need], and average pooling [pool].
    
    Here is a schematic visualization for those unfamiliar with Pytorch:
    
    [eye,, self attend + average <- CNNs,: ViTs <- input
    
    Now, let's go over the specifics of the CNN and ViT. 
    
    Let's overview a Patched Ftheta for any neural network function Ftheta.
    
    Given a grid of arbitrary dimensions, 
    we initialize a uniquely parameterized Ftheta for each designated-size patch
    
    Ftheta1(Patch 1), Ftheta2(Patch 2), ...,
    
    within the grid.
    
    These grid elements are then projected to the output dimensionality of Ftheta, preserving grid ordering.
    
    Unlike convolutions, this operation is intrinsically localized due to parameter non-sharing, 
    and therefore locality embeddings are not needed. 
    However the operation is efficient compared to a fully-connected MLP, much like a CNN.
    
    We can then repeat this for each subsequent grid layer up to chosen depth.
    
    This is the basic variant. We divide the dorsal input-grid into N equal-sized patches along height and width. 
    Each layer is topographically isotropic to the previous.
    
    On the ventral side, a configurable depth-size of residual blocks non-locally transforms the image input.
    Between each block, information is passed to the same-depth layer of the dorsal stream via cross attention. 
    Because this information is one-way, non-locality remains uncompromised. 
    Because of the nature of cross attention, a biological analog of the operation would require signal-passing 
    in both directions to achieve the same effect, which is consistent with observation.
    
    Similarly, a convolutional operation would be expected to have a dense focal point. The ventral V1 region is observed 
    to occupy a smaller region centered around the fovea, 
    compared to the retina-spanning dorsal estuary of the same inputs, consistent with this idea.
    
    We would also expect a more involved routing system to handle convolution via sequencing and spiking 
    rest potential resets. 
    It is indeed observed that the optic nerve stretches out-of-its-way-far from the eye to the back of the head,
    to reach the occipital entrance.
    
    This model reconciles both the what/where-pathway hypothesis, with two disentangled streams,
    and the perceptual/motor-pathway hypothesis, with dorsal-mediated visuo-attention and motor control.
    
    The improved performance is a strong argument for the model. We think the concepts and noted advantages of 
    non-locality, cross-attention with skip connections, and representational disentanglement defined in deep learning
    are pertinent to neuroscience researchers to understanding the technical purpose of the "what"/"where" dichotomy,
    as opposed to other configurations, and disentanglement of perceptual signals, visuo-attention, 
    and motor function in the dorsal region."""


# We can also efficiently substitute the patched MLPs with patched ViTs.

# The operation is linear w.r.t. the number of patches. The ViTs are confined to their respective patches
# and therefore do not infect the total operation with quadratic complexity.
# The cross attentions between streams have the relational advantages of self-attention-based ViTs, but with linear
# time-complexity.

# For that matter, the residual blocks could also be Vision Transformer layers,
# but this would introduce quadratic complexity.

# Can "max-pool" relation-disentanglement style layer by layer using cross attentions and gumbel softmax for escaping
# local hard-attention optima. Can also try to predict dorsal from ventral cross attend as self-supervision.
# Fully architectural. No data augmentation, perturbing, or masking. Debatable whether it can even be called
# self-supervision or if it's just a really good architecture! (Not even locality embeddings)
# Architectural form of this kind of:
# https://medium.com/syncedreview/
# a-leap-forward-in-computer-vision-facebook-ai-says-masked-autoencoders-are-scalable-vision-32c08fadd41f


from Blocks.Architectures.LermanBlocks.BioNet.NonLocality import NonLocalityCNN
from Blocks.Architectures.LermanBlocks.BioNet.Locality import LocalityViT
from Blocks.Architectures.MultiHeadAttention import SelfAttentionBlock


class BioNet(nn.Module):
    """Disentangling "What" And "Where" Pathways In CNNs"""

    def __init__(self, input_shape, out_channels=32, heads=8, depth=3, output_dim=None):
        super().__init__()
        in_channels = input_shape[0]

        # self.ventral_stream = NonLocalityCNN(in_channels, out_channels, depth=depth)
        # self.dorsal_stream = LocalityViT(input_shape, out_channels, depth)
        # self.ventral_stream = CNN(input_shape, out_channels, depth)
        # self.dorsal_stream = CNN(input_shape, out_channels, depth)
        self.ventral_stream = ResNet(input_shape, 3, 2, [64, 64, 128], [2, 2])
        self.dorsal_stream = ResNet(input_shape, 9, 9, [64, 64, 128], [2, 2])

        dims = self.ventral_stream.dims[1:]

        self.cross_talk = nn.ModuleList([CrossAttentionBlock(dim=dim, heads=heads, context_dim=dim)
                                         for dim in dims])

        # self.repr = nn.Sequential(Utils.ChannelSwap(),
        #                           SelfAttentionBlock(dim=dims[-1], heads=heads),
        #                           Utils.ChannelSwap())  # Todo just use einops rearange
        self.repr = nn.Identity()

        self.projection = nn.Identity() if output_dim is None \
            else nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                               nn.Flatten(),
                               nn.Linear(out_channels, 1024),
                               nn.ReLU(inplace=True),
                               nn.Linear(1024, output_dim))

    def repr_shape(self, c, h, w):
        return Utils.cnn_feature_shape(c, h, w, self.dorsal_stream, self.projection)

    def forward(self, input):
        ventral = self.ventral_stream.trunk(input)
        dorsal = self.dorsal_stream.trunk(input)

        t = Utils.ChannelSwap()

        for what, where, talk in zip(self.ventral_stream.ResNet,
                                     self.dorsal_stream.ResNet,
                                     self.cross_talk):
            ventral = what(ventral)
            dorsal = t(talk(t(where(dorsal)),
                            # t(ventral).view(*t(ventral).shape[:-1], 2, -1)))  # Feature redundancy(? till convolved)
                            t(ventral)))

            # if self_supervise:
            #     loss = t(byol(talk2(t(ventral).view(*t(ventral).shape[:-1], 2, -1)), t(dorsal)), t(dorsal).mean(-1))
            #     Utils.optimize(loss,
            #                    self)

        out = self.projection(self.repr(dorsal))
        return out