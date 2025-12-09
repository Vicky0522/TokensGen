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

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.upsampling import *


class Upsample3D(nn.Module):
    """A 2D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=0,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=False,
        interpolate=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate
        conv_cls = nn.Conv3d

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 3
            conv = nn.ConvTranspose3d(
                channels, self.out_channels, kernel_size=kernel_size, stride=(2,1,1), padding=(1,1,1,1,0,0), bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = conv_cls(self.channels, self.out_channels, kernel_size=kernel_size, padding=1, bias=bias)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        #num_chunks: int,
        output_size: Optional[List[int]] = None,
        scale: float = 1.0,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        #print("hidden_states.shape[1]", hidden_states.shape[1], "self.channels", self.channels)
        assert hidden_states.shape[1] == self.channels
        channels, height, width = hidden_states.shape[1:]

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        batch_size = image_only_indicator.shape[0]
        num_frames = hidden_states.shape[0] // batch_size
        hidden_states = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )

        if self.use_conv_transpose:
            raise NotImplementedError
#            hidden_states = F.pad(
#                hidden_states,
#                (0, 0, 0, 0, 1 ,1),
#                'replicate'
#            )
#            hidden_states = hidden_states.reshape(batch_size, channels, num_frames+2, height * width) 
#            hidden_states = F.unfold((num_frames // num_chunks + 2, height * width), padding=0, stride=num_frames // num_chunks)
#            hidden_states = hidden_states.reshape(batch_size, channels, num_frames // num_chunks + 2, height, width, num_chunks).permute(0, 5, 1, 2, 3, 4)
#            hidden_states = hidden_states.reshape(batch_size * num_chunks, channels, num_frames // num_chunks + 2, height, width)
#            hidden_states = self.conv(hidden_states)
#            new_frames_per_chunk, new_height, new_width = hidden_states.shape[-3:]
#            hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
#            hidden_states = hidden_states.reshape(batch_size * num_chunks * new_frames_per_chunk, channels,  new_height, new_width)
#            return hidden_states

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            assert output_size is not None
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                if isinstance(self.conv, LoRACompatibleConv) and not USE_PEFT_BACKEND:
                    hidden_states = self.conv(hidden_states, scale)
                else:
                    hidden_states = self.conv(hidden_states)
            else:
                if isinstance(self.Conv2d_0, LoRACompatibleConv) and not USE_PEFT_BACKEND:
                    hidden_states = self.Conv2d_0(hidden_states, scale)
                else:
                    hidden_states = self.Conv2d_0(hidden_states)

        _, new_height, new_width = hidden_states.shape[-3:]
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        hidden_states = hidden_states.reshape(-1, channels,  new_height, new_width)

        return hidden_states


