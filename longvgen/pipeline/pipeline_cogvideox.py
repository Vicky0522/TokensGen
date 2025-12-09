# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
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

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from einops import rearrange, repeat

import torch
from transformers import AutoImageProcessor, AutoModel
from transformers import T5EncoderModel, T5Tokenizer

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import CogVideoXLoraLoaderMixin
from diffusers.models import AutoencoderKLCogVideoX
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import CogVideoXDDIMScheduler
from diffusers.utils import logging, replace_example_docstring, BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
#from diffusers.pipeline.cogvideo.pipeline_output import CogVideoXPipelineOutput

from longvgen.models.embeddings import (
    get_3d_rotary_pos_embed,
    get_3d_rotary_pos_embed_v2
)
from longvgen.models import CogVideoXTransformer3DModel 
from longvgen.video_ipadapter import Resampler
from longvgen.schedulers import CogVideoXDPMScheduler
from longvgen.fifo_sampling import cogvideo_fifo

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import CogVideoXPipeline
        >>> from diffusers.utils import export_to_video

        >>> # Models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
        >>> pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16).to("cuda")
        >>> prompt = (
        ...     "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
        ...     "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
        ...     "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
        ...     "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
        ...     "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
        ...     "atmosphere of this unique musical performance."
        ... )
        >>> video = pipe(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
        >>> export_to_video(video, "output.mp4", fps=8)
        ```
"""


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device,
         dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out

@dataclass
class CogVideoXPipelineOutput(BaseOutput):

    frames: Union[List[np.ndarray], torch.FloatTensor]
    orig_frames: Union[List[np.ndarray], torch.FloatTensor]
    cache_frames: Union[List[np.ndarray], torch.FloatTensor]

class VideoIPAdapterCogVideoXPipeline(DiffusionPipeline, CogVideoXLoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation using CogVideoX.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. CogVideoX uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`CogVideoXTransformer3DModel`]):
            A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
        resampler: Optional[Resampler] = None,
        image_encoder: Optional[AutoModel] = None,
        feature_extractor: Optional[AutoImageProcessor] = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, 
            text_encoder=text_encoder, 
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            resampler=resampler,
            vae=vae, 
            transformer=transformer, 
            scheduler=scheduler
        )
        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if hasattr(self, "vae") and self.vae is not None else 4
        )
        self.vae_scaling_factor_image = (
            self.vae.config.scaling_factor if hasattr(self, "vae") and self.vae is not None else 0.7
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def encode_image(
        self,
        frames,
        device,
        do_classifier_free_guidance,
        nf_per_chunk,
        resampler_image_rotary_emb,
        resampler_sampling_rotary_emb
    ):
        dtype = next(self.image_encoder.parameters()).dtype

        bsz = frames.shape[0]
        frames = rearrange(frames, "b f c h w -> (b f) c h w")

        frames = _resize_with_antialiasing(frames, (224, 224))
        frames = (frames + 1.0) / 2.0

        # Normalize the image with for CLIP input
        image = self.feature_extractor(
            images=frames.to(torch.float32),
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image, output_hidden_states=True).hidden_states[-2][:,:256]
        image_embeddings = rearrange(image_embeddings, "(b f) n d -> b f n d", b=bsz)

        new_image_embeddings = []
        num_chunks = image_embeddings.shape[1] // nf_per_chunk
        for chunk_id in range(num_chunks):
            new_image_embeddings.append( 
                self.resampler(
                    image_embeddings[:,chunk_id*nf_per_chunk:(chunk_id+1)*nf_per_chunk,...],
                    image_rotary_emb=resampler_image_rotary_emb,
                    sampling_rotary_emb=resampler_sampling_rotary_emb
                )
            )
        image_embeddings = torch.cat(new_image_embeddings, dim=1)
        #image_embeddings = repeat(
        #    image_embeddings,
        #    "b f c h w -> b (ct r) n d", 
        #    r=nf_per_chunk//(image_embeddings.shape[1]//num_chunks)
        #)
        #print(image_embeddings.shape)

        if do_classifier_free_guidance:
            uncond_image = torch.zeros_like(image).to(device=device, dtype=dtype) 
            uncond_image_embeddings = self.image_encoder(uncond_image, output_hidden_states=True).hidden_states[-2][:,:256]
            uncond_image_embeddings = rearrange(uncond_image_embeddings, "(b f) n d -> b f n d", b=bsz)

            new_uncond_image_embeddings = [] 
            for chunk_id in range(num_chunks):
                new_uncond_image_embeddings.append(
                    self.resampler(
                        uncond_image_embeddings[:,chunk_id*nf_per_chunk:(chunk_id+1)*nf_per_chunk,...],
                        image_rotary_emb=resampler_image_rotary_emb,
                        sampling_rotary_emb=resampler_sampling_rotary_emb
                    )
                )
            uncond_image_embeddings = torch.cat(new_uncond_image_embeddings, dim=1)
            #uncond_image_embeddings = repeat(
            #    uncond_image_embeddings,
            #    "b ct n d -> b (ct r) n d",
            #    r=nf_per_chunk//(uncond_image_embeddings.shape[1]//num_chunks)
            #)

            image_embeddings = torch.cat([uncond_image_embeddings, image_embeddings], dim=0)

        return image_embeddings

    def vae_encode_image(
        self,
        frames,
        device,
        do_classifier_free_guidance,
        nf_per_chunk,
        compressed_nf_per_chunk,
        num_chunks,
        resampler_image_rotary_emb,
        resampler_sampling_rotary_emb,
        image_embeddings = None,
    ):
        dtype = next(self.vae.parameters()).dtype

        def encode_video(video):
            video = video.to(device=device, dtype=dtype) 
            video = rearrange(video, "b f c h w -> b c f h w")
            # padding one chunk
            video = torch.cat([video,] + [video[:,:,[-1]],]*nf_per_chunk, dim=2)
            latents_list = []
            num_chunks = video.shape[2] // nf_per_chunk
            for chunk_id in range(num_chunks):
                latents = self.vae.encode(video[:,:,chunk_id*nf_per_chunk:(chunk_id+1)*nf_per_chunk]).latent_dist.sample() * self.vae.config.scaling_factor
                latents_list.append(latents)
            latents = torch.cat(latents_list, dim=2)
            latents = rearrange(latents, "b c f h w -> b f c h w")
            return latents

        if image_embeddings is None:

            image_embeddings = encode_video(frames)
            bsz = image_embeddings.shape[0]
            image_embeddings = rearrange(image_embeddings, "b f c h w -> (b f) c h w")
            image_embeddings = self.transformer.patch_embed.proj(image_embeddings) 
            image_embeddings = rearrange(image_embeddings, "(b f) c h w -> b f (h w) c", b=bsz)

            new_image_embeddings = []
            _num_chunks = image_embeddings.shape[1] // compressed_nf_per_chunk
            for chunk_id in range(_num_chunks):
                new_image_embeddings.append( 
                    self.resampler(
                        image_embeddings[:,chunk_id*compressed_nf_per_chunk:(chunk_id+1)*compressed_nf_per_chunk,...],
                        image_rotary_emb=resampler_image_rotary_emb,
                        sampling_rotary_emb=resampler_sampling_rotary_emb
                    )
                )
            image_embeddings = torch.cat(new_image_embeddings, dim=1)

        else:
            
            image_embeddings = torch.cat(
                [image_embeddings,] + [image_embeddings[:,[-1]],] * (image_embeddings.shape[1]//num_chunks),
                dim=1
            )

        if do_classifier_free_guidance:
            uncond_frames = torch.zeros((image_embeddings.shape[0], nf_per_chunk * num_chunks, 3, 480, 720))
            uncond_image_embeddings = encode_video(uncond_frames)
            uncond_image_embeddings = rearrange(uncond_image_embeddings, "b f c h w -> (b f) c h w")
            uncond_image_embeddings = self.transformer.patch_embed.proj(uncond_image_embeddings) 
            uncond_image_embeddings = rearrange(uncond_image_embeddings, "(b f) c h w -> b f (h w) c", b=image_embeddings.shape[0])

            new_uncond_image_embeddings = [] 
            for chunk_id in range(num_chunks+1):
                new_uncond_image_embeddings.append(
                    self.resampler(
                        uncond_image_embeddings[:,chunk_id*compressed_nf_per_chunk:(chunk_id+1)*compressed_nf_per_chunk,...],
                        image_rotary_emb=resampler_image_rotary_emb,
                        sampling_rotary_emb=resampler_sampling_rotary_emb
                    )
                )
            uncond_image_embeddings = torch.cat(new_uncond_image_embeddings, dim=1)

            # if not equal, padding frames to the last
            if image_embeddings.shape[1] != uncond_image_embeddings.shape[1]:
                image_embeddings = torch.cat(
                    [image_embeddings] + [image_embeddings[:,[-1]],]*(uncond_image_embeddings.shape[1]-image_embeddings.shape[1]),
                    dim=1
                )
            #image_embeddings = torch.cat([uncond_image_embeddings, image_embeddings], dim=0)
            image_embeddings = torch.cat([image_embeddings, image_embeddings], dim=0)

        return image_embeddings

    def prepare_latents(
        self, batch_size, num_channels_latents, num_chunks, num_frames_per_chunk, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_chunks * ((num_frames_per_chunk - 1) // self.vae_scale_factor_temporal + 1),
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents: torch.Tensor, nf_per_chunk=13) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae_scaling_factor_image * latents

        frames = []
        for chunk_id in range(latents.shape[2] // nf_per_chunk):
            frames.append(self.vae.decode(latents[:,:,chunk_id*nf_per_chunk:(chunk_id+1)*nf_per_chunk]).sample)
        frames = torch.cat(frames, dim=2)
        return frames

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def fuse_qkv_projections(self) -> None:
        r"""Enables fused QKV projections."""
        self.fusing_transformer = True
        self.transformer.fuse_qkv_projections()

    def unfuse_qkv_projections(self) -> None:
        r"""Disable QKV projection fusion if enabled."""
        if not self.fusing_transformer:
            logger.warning("The Transformer was not initially fused for QKV projections. Doing nothing.")
        else:
            self.transformer.unfuse_qkv_projections()
            self.fusing_transformer = False

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        base_size_width = 720 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        base_size_height = 480 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        grid_crops_coords = list(grid_crops_coords)
        grid_crops_coords[0] = [0,] + list(grid_crops_coords[0])
        grid_crops_coords[1] = [num_frames,] + list(grid_crops_coords[1])
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.transformer.config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(num_frames, grid_height, grid_width),
        )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin

    def _prepare_vip_rotary_positional_embeddings(
        self,
        grid_t,
        grid_h,
        grid_w,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed_v2(
            embed_dim=self.transformer.config.attention_head_dim,
            grid_t = grid_t,
            grid_h = grid_h,
            grid_w = grid_w
        )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        frames: Optional[torch.FloatTensor] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        num_frames_per_chunk: int = 49,
        max_num_chunks: Optional[int] = 1,
        max_num_chunks_w_fifo: Optional[int] = None,
        max_num_chunks_wo_fifo: Optional[int] = 1,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        decode_chunk_size: Optional[int] = None,
        vip_scale: float = 1.0,
        sampling_mode: str = None,
        sampling_params: Dict[str, Any] = None,
        cache_idx: Optional[List[int]] = [],
        video_ipadapter_start_frame_idx: Optional[int] = 1000,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 4. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        nf_per_chunk = num_frames_per_chunk
        num_chunks = frames.shape[1]//nf_per_chunk if frames is not None else max_num_chunks
        num_frames = frames.shape[1] if frames is not None else max_num_chunks * nf_per_chunk
        num_chunks_wo_fifo = max(1, min(max_num_chunks_wo_fifo, num_chunks))
        num_chunks_w_fifo = max(1, min(max_num_chunks_w_fifo, num_chunks)) if max_num_chunks_w_fifo is not None else num_chunks
        if sampling_mode is not None and "freeinit" in sampling_mode:
            num_chunks_wo_fifo = num_chunks
        cache_idx = [] if cache_idx is None else cache_idx
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
        compressed_nf_per_chunk = (nf_per_chunk - 1) // self.vae_scale_factor_temporal + 1
        
        use_vip = (self.resampler is not None)
        use_vae_as_encoder = (use_vip and self.image_encoder is None)
        use_fifo = (sampling_mode is not None)

        num_videos_per_prompt = 1

        if num_frames_per_chunk > 49:
            raise ValueError(
                "The number of frames must be less than 49 for now due to static positional embeddings. This will be updated in the future to remove this limitation."
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # set vip scale
        for fullname, module in self.transformer.named_modules():
            if module.__class__.__name__ == "VideoIPAdapterCogVideoXAttnProcessor2_0":
                module.scale = vip_scale

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)


        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_chunks_wo_fifo,
            nf_per_chunk,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        random_latents = latents.clone()

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, compressed_nf_per_chunk, device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )
        if use_vip:
            vip_image_rotary_grid_h = np.linspace(
                0, 
                latents.shape[-2]//self.transformer.config.patch_size,
                latents.shape[-2]//self.transformer.config.patch_size,
                endpoint=False, dtype=np.float32
            )
            vip_image_rotary_grid_w = np.linspace(
                0,
                latents.shape[-1]//self.transformer.config.patch_size,
                latents.shape[-1]//self.transformer.config.patch_size,
                endpoint=False, dtype=np.float32
            )
            vip_image_rotary_grid_t = np.linspace(
                0,
                num_chunks * compressed_nf_per_chunk,
                num_chunks * compressed_nf_per_chunk,
                endpoint=False, dtype=np.float32
            )
            print("vip_image_rotary_grid_t", vip_image_rotary_grid_t)
            vip_nf_per_chunk = self.resampler.config.num_temporal_queries
            vip_condition_rotary_grid_h = np.linspace(
                0,
                latents.shape[-2]//self.transformer.config.patch_size,
                self.resampler.config.num_height_queries,
                endpoint=False, dtype=np.float32
            )
            vip_condition_rotary_grid_w = np.linspace(
                0,
                latents.shape[-1]//self.transformer.config.patch_size,
                self.resampler.config.num_width_queries,
                endpoint=False, dtype=np.float32
            )
            vip_condition_rotary_grid_t = np.concatenate(
                [np.linspace(
                    video_ipadapter_start_frame_idx + i*compressed_nf_per_chunk, 
                    video_ipadapter_start_frame_idx + (i+1)*compressed_nf_per_chunk,
                    vip_nf_per_chunk,
                    endpoint=False, dtype=np.float32
                 ) for i in range(num_chunks+1)
                ]
            )
            print("vip_condition_rotary_grid_t", vip_condition_rotary_grid_t)
            resampler_image_rotary_emb = (
                self._prepare_vip_rotary_positional_embeddings(
                    grid_t = np.linspace(
                        0,
                        self.resampler.config.max_temporal_seq_len,
                        self.resampler.config.max_temporal_seq_len,
                        endpoint=False, dtype=np.float32
                    ),
                    grid_h = np.linspace(
                        0,
                        self.resampler.config.max_height_seq_len,
                        self.resampler.config.max_height_seq_len,
                        endpoint=False, dtype=np.float32
                    ),
                    grid_w = np.linspace(
                        0,
                        self.resampler.config.max_width_seq_len,
                        self.resampler.config.max_width_seq_len,
                        endpoint=False, dtype=np.float32
                    ),
                    device=device,
                )
            )
            resampler_sampling_rotary_emb = (
                self._prepare_vip_rotary_positional_embeddings(
                    grid_t = np.linspace(
                        video_ipadapter_start_frame_idx,
                        video_ipadapter_start_frame_idx + self.resampler.config.max_temporal_seq_len,
                        self.resampler.config.num_temporal_queries,
                        endpoint=False, dtype=np.float32
                    ),
                    grid_h = np.linspace(
                        0,
                        self.resampler.config.max_height_seq_len,
                        self.resampler.config.num_height_queries,
                        endpoint=False, dtype=np.float32
                    ),
                    grid_w = np.linspace(
                        0,
                        self.resampler.config.max_width_seq_len,
                        self.resampler.config.num_width_queries,
                        endpoint=False, dtype=np.float32
                    ),
                    device=device,
                )
            )
        # 7.5 Encode input image
        if use_vip:
            if use_vae_as_encoder: 
                image_embeddings = self.vae_encode_image(
                    frames,
                    device,
                    do_classifier_free_guidance,
                    nf_per_chunk,
                    compressed_nf_per_chunk,
                    num_chunks,
                    resampler_image_rotary_emb,
                    resampler_sampling_rotary_emb,
                    image_embeddings = image_embeddings,
                )
            else:
                image_embeddings = self.encode_image(
                    frames,
                    device,
                    do_classifier_free_guidance,
                    nf_per_chunk,
                    resampler_image_rotary_emb,
                    resampler_sampling_rotary_emb
                )
            print("image_embeddings.shape", image_embeddings.shape)
            print(image_embeddings.min(), image_embeddings.max(), image_embeddings.mean())

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        if use_fifo and "fifo" in sampling_mode:
            fifo_latents = []
            fifo_old_pred_original_sample = []

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if use_fifo and "fifo" in sampling_mode:
                    #print(latents.shape, latents.min(), latents.max(), latents.mean())
                    fifo_latents = [latents[:,[max(0,compressed_nf_per_chunk-1-i)]],] + fifo_latents
                    if old_pred_original_sample is None:
                        fifo_old_pred_original_sample = [None,] + fifo_old_pred_original_sample
                    else:
                        fifo_old_pred_original_sample = [old_pred_original_sample[:,[max(0,compressed_nf_per_chunk-1-i)]],] + fifo_old_pred_original_sample

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                all_noise_preds = []

                for chunk_id in range(num_chunks_wo_fifo):
                    progress_bar.set_description(f"Processing step {t} chunk_id {chunk_id}")
                    if use_vip:
                        #_chunk_id = chunk_id
                        #chunk_id = 0
                        # recompute rotary embeddings
                        input_vip_condition_rotary_emb = self._prepare_vip_rotary_positional_embeddings(
                            grid_t=vip_condition_rotary_grid_t[chunk_id*vip_nf_per_chunk:(chunk_id*vip_nf_per_chunk+min(vip_nf_per_chunk+1, compressed_nf_per_chunk))],
                            grid_h=vip_condition_rotary_grid_h,
                            grid_w=vip_condition_rotary_grid_w,
                            device=device,
                        )
                        input_vip_image_rotary_emb = self._prepare_vip_rotary_positional_embeddings(
                            grid_t=vip_image_rotary_grid_t[chunk_id*compressed_nf_per_chunk:(chunk_id+1)*compressed_nf_per_chunk],
                            grid_h=vip_image_rotary_grid_h,
                            grid_w=vip_image_rotary_grid_w,
                            device=device,
                        )
                        if not self.transformer.config.use_rotary_positional_embeddings:
                            relative_vip_grid_t = vip_condition_rotary_grid_t[chunk_id*vip_nf_per_chunk:(chunk_id*vip_nf_per_chunk+min(vip_nf_per_chunk+1, compressed_nf_per_chunk))] - vip_image_rotary_grid_t[chunk_id*compressed_nf_per_chunk]
                            relative_vip_grid_t = relative_vip_grid_t[None,...]
                            print("relative_vip_grid_t", relative_vip_grid_t)
                        else:
                            relative_vip_grid_t = None
                        #chunk_id = _chunk_id

                    # predict noise model_output
                    #print("latent_model_input.shape", latent_model_input.shape)
                    #print("1")
                    #print(latent_model_input[:,chunk_id*compressed_nf_per_chunk:(chunk_id+1)*compressed_nf_per_chunk].shape)
                    #print("chunk_id", chunk_id)
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input[:,chunk_id*compressed_nf_per_chunk:(chunk_id+1)*compressed_nf_per_chunk],
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        image_rotary_emb=image_rotary_emb,
                        vip_image_rotary_emb=input_vip_image_rotary_emb if use_vip else None,
                        vip_condition_rotary_emb=input_vip_condition_rotary_emb if use_vip else None,
                        vip_grid_t = relative_vip_grid_t if use_vip else None,
                        vip_encoder_hidden_states=image_embeddings[:,chunk_id*vip_nf_per_chunk:(chunk_id*vip_nf_per_chunk+min(vip_nf_per_chunk+1, compressed_nf_per_chunk))] if use_vip else None,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred.float()
                    all_noise_preds.append(noise_pred)

                noise_pred = torch.cat(all_noise_preds, dim=1)

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred, 
                        t, 
                        prev_t,
                        latents, 
                        **extra_step_kwargs, 
                        return_dict=False
                    )
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        prev_t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)


                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        orig_latents = latents.clone()

        use_fifo = True
        if sampling_mode == "fifo":
            sampling_func = cogvideo_fifo
            latents = latents[:,:compressed_nf_per_chunk]
        elif sampling_mode == "denoising_together":
            sampling_func = cogvideo_denoising_together
            latents = random_latents[:,:,:compressed_nf_per_chunk]
        elif sampling_mode == "fifo_freeinit":
            sampling_func = cogvideo_fifo_freeinit
        elif sampling_mode == None:
            use_fifo = False
        else:
            raise NotImplementedError

        if use_fifo:
            print("image_embeddings", image_embeddings.mean(), image_embeddings.min(), image_embeddings.max())
            print("fifo_latents", [item.mean() for item in fifo_latents])
            print("fifo_old_pred_original_sample", [item.mean() for item in fifo_old_pred_original_sample if item is not None])
            latents, cache_latents = sampling_func(
                self,
                fifo_latents=torch.cat(fifo_latents, dim=1), # length: num_inference_timesteps
                fifo_old_pred_original_sample=fifo_old_pred_original_sample,
                nf_per_chunk=compressed_nf_per_chunk,
                vip_nf_per_chunk=vip_nf_per_chunk if use_vip else None,
                num_frames=num_chunks_w_fifo*compressed_nf_per_chunk,
                image_embeddings=image_embeddings, # length: num_chunks * compressed_nf_per_chunk
                timesteps=timesteps,
                num_inference_steps=num_inference_steps,
                do_classifier_free_guidance=do_classifier_free_guidance,
                use_dynamic_cfg=use_dynamic_cfg,
                prompt_embeds=prompt_embeds,
                image_rotary_emb=image_rotary_emb,
                vip_image_rotary_grid=[
                    vip_image_rotary_grid_t,
                    vip_image_rotary_grid_h,
                    vip_image_rotary_grid_w,
                ] if use_vip else None, # length: num_chunks * compressed_nf_per_chunk
                vip_condition_rotary_grid=[
                    vip_condition_rotary_grid_t,
                    vip_condition_rotary_grid_h,
                    vip_condition_rotary_grid_w,
                ] if use_vip else None, # length: num_chunks * vip_nf_per_chunk
                attention_kwargs=attention_kwargs,
                guidance_scale=guidance_scale,
                extra_step_kwargs=extra_step_kwargs,
                cache_idx=cache_idx,
                condition_frames=frames,
                video_ipadapter_start_frame_idx=video_ipadapter_start_frame_idx,
                device=device,
                **sampling_params
            )
        else:
            latents, cache_latents = None, []

        for i in range(len(cache_latents)):
            cache_latents[i] = torch.cat(cache_latents[i], dim=1)

        if not output_type == "latent":
            if latents is not None:
                video = self.decode_latents(latents)
                video = self.video_processor.postprocess_video(video=video, output_type=output_type)
            else:
                video = None
            orig_video = self.decode_latents(orig_latents)
            orig_video = self.video_processor.postprocess_video(video=orig_video, output_type=output_type)
            cache_video = []
            for i in range(len(cache_latents)):
                cache_video.append(
                    self.video_processor.postprocess_video(
                        video = self.decode_latents(cache_latents[i]),
                        output_type=output_type
                    )
                )
        else:
            video = latents
            orig_video = orig_latents
            cache_video = cache_latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (orig_video, video, cache_video)

        return CogVideoXPipelineOutput(frames=video, orig_frames=orig_video, cache_frames=cache_video)
