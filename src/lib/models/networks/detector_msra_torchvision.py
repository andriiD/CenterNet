# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torchvision.models as models

BN_MOMENTUM = 0.1


class PoseResNet(nn.Module):

    def __init__(self, backbone, heads, head_conv, **kwargs):
        """
        Supported backbones: all resnets, mobilenet_v2 from torchvision

        :param backbone: mobilenet_v2, resnet18, resnet*
        :param heads:
        :param head_conv:
        :param kwargs:
        """
        super(PoseResNet, self).__init__()
        self.deconv_with_bias = False
        self.heads = heads

        # Set up the detector's backbone from torchvision.models
        base_net = getattr(models, backbone)(pretrained=True)
        print(base_net)
        if backbone.startswith("resnet"):
            self.inplanes = base_net.fc.in_features  # Size of the features before heads
            base_net = list(base_net.children())
            self.encoder = nn.Sequential(*base_net[:-2])  # Exclude original fully connected and pooling layers
        elif backbone.startswith("mobilenet"):
            self.inplanes = base_net.classifier[1].in_features  # Size of the features before heads
            base_net = list(base_net.children())
            self.encoder = nn.Sequential(*base_net[:-1])  # Exclude last sequential
        else:
            raise ValueError("wrong backbone name prided", backbone)

        #
        # print(self.encoder)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        # Used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 256, 256],
            [4, 4, 4],
        )
        print(self.deconv_layers)
        for head in sorted(self.heads):
            print("head", head)
            num_output = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(256, head_conv,
                      kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, num_output,
                      kernel_size=1, stride=1, padding=0))
            else:
                fc = nn.Conv2d(
                    in_channels=256,
                    out_channels=num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            self.__setattr__(head, fc)

        self.init_weights()

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)  # batch_size x self.inplanes x 7 x 7

        x = self.deconv_layers(x)  # batch_size x 256 x 128 x 128
        # print("\nAfter deconv:", x.size())
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

    def init_weights(self):
        print('=> init resnet deconv weights from normal distribution')
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # print('=> init {}.weight as 1'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # print('=> init final conv weights from normal distribution')
        for head in self.heads:
          final_layer = self.__getattr__(head)
          for i, m in enumerate(final_layer.modules()):
              if isinstance(m, nn.Conv2d):
                  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                  # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                  # print('=> init {}.bias as 0'.format(name))
                  if m.weight.shape[0] == self.heads[head]:
                      if 'hm' in head:
                          nn.init.constant_(m.bias, -2.19)
                      else:
                          nn.init.normal_(m.weight, std=0.001)
                          nn.init.constant_(m.bias, 0)



def get_pose_net(backbone, heads, head_conv):
  model = PoseResNet(backbone, heads, head_conv=head_conv)
  return model
