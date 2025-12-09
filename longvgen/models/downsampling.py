# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.downsampling import *

class Downsample3D(nn.Module):
    """A 3D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 0,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        #conv_cls = nn.Conv2d if USE_PEFT_BACKEND else LoRACompatibleConv
        conv_cls = nn.Conv3d

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv = conv_cls(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv3d = conv

    def forward(
        self, 
        hidden_states: torch.FloatTensor, 
        num_chunks: int,
        scale: float = 1.0,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        #if self.use_conv and self.padding == 0:
        #    pad = (0, 1, 0, 1)
        #    hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels
        channels, height, width = hidden_states.shape[1:]
        batch_size, num_frames = image_only_indicator.shape
        hidden_states = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )
        hidden_states = F.pad(
            hidden_states,
            (0, 0, 0, 0, 1 ,1),
            'replicate'
        )
#        hidden_states = hidden_states.reshape(batch_size, channels, num_frames+2, height * width) 
#        hidden_states = F.unfold(
#            hidden_states, 
#            (num_frames // num_chunks + 2, height * width), 
#            padding=0, stride=num_frames // num_chunks
#        )
#        hidden_states = hidden_states.reshape(batch_size, channels, num_frames // num_chunks + 2, height, width, num_chunks).permute(0, 5, 1, 2, 3, 4)
#        hidden_states = hidden_states.reshape(batch_size * num_chunks, channels, num_frames // num_chunks + 2, height, width)
        hidden_states = F.pad(
            hidden_states,
            (1, 1, 1, 1, 0, 0),
            'constant'
        )

        hidden_states = self.conv3d(hidden_states)
        new_num_frames, new_height, new_width = hidden_states.shape[-3:]
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        hidden_states = hidden_states.reshape(batch_size * new_num_frames, channels, new_height, new_width)

        return hidden_states

