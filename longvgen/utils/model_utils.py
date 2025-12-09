import argparse
import datetime
import logging
import inspect
import math
import os
import random
import gc
import copy
import imageio
import numpy as np
import PIL
from PIL import Image
from scipy.stats import anderson
from typing import Dict, Optional, Tuple, Callable
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers.models.clip.modeling_clip import CLIPEncoder

import diffusers
from diffusers.models import AutoencoderKL
from diffusers import DDIMScheduler, TextToVideoSDPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.unet_3d_blocks import \
            (CrossAttnDownBlockSpatioTemporal, 
             DownBlockSpatioTemporal,
             CrossAttnUpBlockSpatioTemporal,
             UpBlockSpatioTemporal)
from einops import rearrange, repeat

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteSchedulerOutput, EulerDiscreteScheduler 

def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    # Based on:
    # https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78

    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    return outputs

@torch.no_grad()
def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents

def load_primary_models(pretrained_model_path):
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        pretrained_model_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(
        pretrained_model_path, subfolder="feature_extractor", revision=None
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_path, subfolder="image_encoder", revision=None, variant="fp16"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pretrained_model_path, subfolder="vae", revision=None, variant="fp16")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        pretrained_model_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )

    return noise_scheduler, feature_extractor, image_encoder, vae, unet

def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0())

def is_attn(name):
    return ('attn1' or 'attn2' == name.split('.')[-1])

def set_torch_2_attn(unet):
    optim_count = 0

    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0:
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")

def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet):
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn

        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if enable_torch_2:
            set_torch_2_attn(unet)

    except Exception as e:
        print(e)
        print("Could not enable memory efficient attention for xformers or Torch 2.0.")


class LongVGenEulerDiscreteScheduler(EulerDiscreteScheduler):

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        conditional_latents: torch.FloatTensor = None,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[EulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        noise = randn_tensor(
            model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
        )

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        if self.config.prediction_type == "original_sample" or self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
        elif self.config.prediction_type == "v_prediction":
            # denoised = model_output * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )
        
        #for controller in self.controller:
        #    pred_original_sample = controller.step_callback(pred_original_sample)
        # color preservation
        #color_delta = torch.mean(conditional_latents[:,0:1,:,:,:]) - torch.mean(pred_original_sample[:,0:1,:,:,:])
        #print("color_delta", color_delta)
        #pred_original_sample = pred_original_sample + color_delta
        
        

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_hat

        dt = self.sigmas[self.step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        #self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return EulerDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def step_callback(self):
        self._step_index += 1 

@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    r"""
    Output class for zero-shot text-to-video pipeline.

    Args:
        frames (`[List[PIL.Image.Image]`, `np.ndarray`]):
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    frames: Union[List[PIL.Image.Image], np.ndarray]
    latents: torch.Tensor

class LongVGenStableVideoDiffusionPipeline(StableVideoDiffusionPipeline):

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        image_latents: torch.Tensor = None
    ):
        if image_latents is None:
            image = image.to(device=device)
            image_latents = self.vae.encode(image).latent_dist.mode()
        else:
            image_latents = rearrange(image_latents, "b f c h w -> (b f) c h w")

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents
    
    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        edited_firstframe: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor] = None,
        image_latents: torch.FloatTensor = None,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_frames_per_chunk: Optional[int] = None,
        overlay_size: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 2.5,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`int`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        num_frames_per_chunk = num_frames_per_chunk if num_frames_per_chunk is not None else self.unet.config.num_frames
        overlay_size = overlay_size if overlay_size is not None else num_frames_per_chunk // 2 
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames_per_chunk

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        edited_firstframe = image if edited_firstframe is None else edited_firstframe 
        image_embeddings = self._encode_image(edited_firstframe, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        print("image_embeddings.shape", image_embeddings.shape)

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        image = self.image_processor.preprocess(image, height=height, width=width)
        edited_firstframe = self.image_processor.preprocess(edited_firstframe, height=height, width=width)
        print("before vae", image.min(), image.max())
        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype)
        image = image + noise_aug_strength * noise
        edited_firstframe = edited_firstframe + noise_aug_strength * noise
        if image_latents is not None:
            noise_tmp = randn_tensor(image_latents.shape, generator=generator, device=image_latents.device, dtype=image_latents.dtype)
            image_latents = image_latents / self.vae.config.scaling_factor 
            image_latents = image_latents + noise_aug_strength * noise_tmp

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        print("before vae", image.min(), image.max())
        image_latents = self._encode_vae_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance, image_latents = image_latents)            
        firstframe_latents = self._encode_vae_image(edited_firstframe, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        #noise = randn_tensor(image_latents.shape, generator=generator, device=image_latents.device, dtype=image_latents.dtype)
        #image_latents = image_latents + noise_aug_strength * noise / self.vae.config.scaling_factor
        #firstframe_latents = firstframe_latents + noise_aug_strength * noise / self.vae.config.scaling_factor

        image_latents = image_latents.to(image_embeddings.dtype)
        firstframe_latents = firstframe_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]

        skip=1
        image_latents = torch.cat(
            [
             image_latents.unsqueeze(1).repeat(1, skip, 1, 1, 1),
             firstframe_latents.unsqueeze(1).repeat(1, num_frames-skip, 1, 1, 1)
            ],
            dim=1
        )

        #image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        print("image latents", image_latents.min(), image_latents.max())

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        previous_latents = latents.clone().to(latents.device) 
        current_latents = latents.clone().to(latents.device) 
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                begin_id = 0
                while begin_id < num_frames:
                    begin_id = max(0, begin_id - overlay_size)
                    end_id = min(begin_id + num_frames_per_chunk, num_frames)
                    print(begin_id, end_id)
                    latents = previous_latents[:,begin_id:end_id,...]
                    print(latents.shape)
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # Concatenate image_latents over channels dimention
                    latent_model_input = torch.cat(
                        [latent_model_input, image_latents[:,begin_id:end_id,...]], 
                        dim=2
                    )

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=image_embeddings,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    guidance_scale = self.guidance_scale[:,begin_id:end_id,...]

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    #conditional_latents = image_latents.chunk(2)[1] if self.do_classifier_free_guidance else image_latents 
                    #latents = self.scheduler.step(noise_pred, t, latents, conditional_latents=conditional_latents*self.vae.config.scaling_factor).prev_sample
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                    if i % 2 == 0:
                        if begin_id == 0:
                            current_latents[:,begin_id:end_id,...] = latents
                        else:
                            current_latents[:,begin_id+overlay_size:end_id,...] = latents[:,overlay_size:,...]
                    else:
                        if end_id == num_frames:
                            current_latents[:,begin_id:end_id,...] = latents
                        else:
                            current_latents[:,begin_id:end_id-overlay_size,...] = latents[:,begin_id:end_id-overlay_size,...]
                        
                    begin_id = end_id


                previous_latents = current_latents
                current_latents = previous_latents.clone().to(previous_latents.device)
                self.scheduler.step_callback()

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    previous_latents = callback_outputs.pop("latents", previous_latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(current_latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = current_latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames, latents=current_latents)
