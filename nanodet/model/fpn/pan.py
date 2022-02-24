# Modification 2020 RangiLyu
# Copyright 2018-2019 Open-MMLab.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn.functional as F

from .fpn import FPN

import torch
import torch.nn as nn
from ..module.conv import ConvModule, DepthwiseConvModule


class PAN(FPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        activation (str): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        conv_cfg=None,
        # norm_cfg=None,
        norm_cfg=dict(type="BN"),
        activation=None,
        kernel_size=5,
    ):
        super(PAN, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            start_level,
            end_level,
            conv_cfg,
            norm_cfg,
            activation,
        )
        self.init_weights()
        # 为保证亿智模型转换前后Bin文件一致，需替换上采样层为反卷积
        self.deconvs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for idx in range(len(self.in_channels) + 1):
            self.deconvs.append(nn.ConvTranspose2d(self.out_channels, self.out_channels, 2, 2))
            # self.convs.append(DepthwiseConvModule(
            #         self.out_channels * 2,
            #         self.out_channels,
            #         3,
            #         stride=1,
            #         padding=1,
            #         norm_cfg=norm_cfg,
            #         activation=activation,
            #     ))
        self.downsamples = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                DepthwiseConvModule(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            )

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode="bilinear")
            laterals[i - 1] = laterals[i - 1] + self.deconvs[i](laterals[i])  # 为保证亿智模型转换前后Bin文件一致，需替换上采样层为反卷积
            # laterals[i - 1] = laterals[i - 1] + self.deconvs[i](self.deconvs[i](laterals[i]))  # 由20*12经两次反卷积得到80*48
            # concat + conv
            # upsample_feat = self.deconvs[i](laterals[i])
            # laterals[i - 1] = self.convs[i](torch.cat([upsample_feat, laterals[i - 1]], 1))


        # build outputs
        # part 1: from original levels
        inter_outs = [laterals[i] for i in range(used_backbone_levels)]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            # inter_outs[i + 1] += F.interpolate(inter_outs[i], scale_factor=0.5, mode="bilinear")
            # inter_outs[i + 1] += F.interpolate(inter_outs[i], scale_factor=0.25, mode="bilinear")
            inter_outs[i + 1] = inter_outs[i + 1] + self.downsamples[i](inter_outs[i])
            # inter_outs[i + 1] = inter_outs[i + 1] + self.downsamples[i](self.downsamples[i](inter_outs[i]))  # 由20*12经两次反卷积得到80*48
            # concat + conv
            # downsample_feat = self.downsamples[i](inter_outs[i])
            # inter_outs[i + 1] = self.convs[i](torch.cat([downsample_feat, inter_outs[i + 1]], 1))

        outs = []
        outs.append(inter_outs[0])
        outs.extend([inter_outs[i] for i in range(1, used_backbone_levels)])

        # extra layers
        # outs.append(self.downsamples[0](laterals[-1]) + self.downsamples[0](outs[-1]))
        # outs.append(self.downsamples[0](self.downsamples[0](laterals[-1])) + self.downsamples[0](self.downsamples[0](outs[-1])))
        # outs.append(F.interpolate(laterals[-1], scale_factor=0.25, mode="bilinear") + F.interpolate(outs[-1], scale_factor=0.25, mode="bilinear"))

        return tuple(outs)
