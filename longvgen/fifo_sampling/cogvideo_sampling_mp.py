import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from einops import rearrange, repeat
from tqdm.auto import tqdm
from tqdm import trange
from tqdm.auto import tqdm
import copy
import math

import numpy as np
import torch

from accelerate.utils import (
    gather_object,
)

from longvgen.schedulers import CogVideoXDPMScheduler

def cogvideo_fifo_mp(
    pipe,
    accelerator,
    fifo_latents,
    fifo_old_pred_original_sample,
    nf_per_chunk,
    vip_nf_per_chunk,
    num_frames,
    timesteps,
    num_inference_steps,
    do_classifier_free_guidance,
    use_dynamic_cfg,
    prompt_embeds,
    image_rotary_emb,
    vip_image_rotary_grid,
    vip_condition_rotary_grid,
    attention_kwargs,
    guidance_scale,
    extra_step_kwargs,
    image_embeddings: Optional[torch.Tensor] = None,
    cache_idx: Optional[List[int]] = [],
    condition_frames: Optional[torch.Tensor] = None,
    num_partitions: Optional[int] = 4,
    use_adaptive_padding: Optional[bool] = False,
    use_sliding_window_embedding: Optional[bool] = False,
    video_ipadapter_start_frame_idx: Optional[int] = 1000,
    device: torch.device = None,
    **kwargs
):
    lookahead_denoising = True
    if use_sliding_window_embedding:
        assert condition_frames is not None
    use_vip = (image_embeddings is not None)

    l_nf_per_chunk = nf_per_chunk-nf_per_chunk//2
    r_nf_per_chunk = nf_per_chunk // 2

    def prepare_fifo_latents():
        latents_list = []
        if lookahead_denoising:
            for i in range(r_nf_per_chunk):
                latents_list.append( 
                        fifo_latents[:,[0]]
                )

        latents_list = latents_list + [fifo_latents,]

        return torch.cat(latents_list, dim=1)

    def prepare_fifo_vip_image_rotary_grid_t(grid_t):
        grid_t_list = [grid_t[[0]],] * (r_nf_per_chunk + num_inference_steps-nf_per_chunk)
        grid_t_list += [grid_t[:nf_per_chunk],]

        initial_grid_t = np.concatenate(grid_t_list)

        grid_t_queue = [grid_t[nf_per_chunk:],] + [np.linspace(grid_t[-1]+1, grid_t[-1]+1+num_inference_steps, num_inference_steps, endpoint=False, dtype=np.float32)]
        grid_t_queue = np.concatenate(grid_t_queue)
        
        return initial_grid_t, grid_t_queue

    def prepare_fifo_vip_condition_rotary_grid_t(grid_t):
        grid_t_list = [grid_t,]
        for i in range(num_inference_steps // nf_per_chunk + 1):
            grid_t_list += [grid_t[-vip_nf_per_chunk:] + (i+1)*nf_per_chunk] 
        return np.concatenate(grid_t_list)
                
    def prepare_fifo_image_embeddings(): 
        embeddings_list = [image_embeddings]

        embeddings_list += [image_embeddings[:,-vip_nf_per_chunk:],] * (num_inference_steps // nf_per_chunk + 1)
        
        _embeddings = torch.cat(embeddings_list, dim=1)

        return _embeddings 

    def find_embed_index(vip_condition_rotary_grid_t, start_image_pos_idx):
        return np.searchsorted(
            vip_condition_rotary_grid_t,
            start_image_pos_idx + video_ipadapter_start_frame_idx,
            side='right',
        ) - 1

#    def prepare_fifo_cond_frames():
#        frames_list = []
#        if lookahead_denoising:
#            frames_list += [torch.cat([condition_frames[:,[0]],] * (nf_per_chunk//2), dim=1),]
#
#        frames_list += [torch.cat([condition_frames[:,[0]],] * (num_inference_steps-nf_per_chunk), dim=1),]
#        frames_list += [condition_frames[:,:nf_per_chunk],]
#
#        initial_frames = torch.cat(frames_list, dim=1)
#
#        frames_queue = torch.cat(
#            [condition_frames[:,nf_per_chunk:],] + \
#            [torch.cat([condition_frames[:,[-1]],] * num_inference_steps, dim=1),],
#            dim = 1
#        )
#
#        return initial_frames, frames_queue

    def shift_latents(_latents, _old_pred_original_sample):
        # shift latents
        _latents[:,:-1] = _latents[:,1:].clone()
        _old_pred_original_sample[:-1] = copy.deepcopy(_old_pred_original_sample[1:])

        # add new noise to the last frame
        _latents[:,-1] = torch.randn_like(_latents[:,-1])
        _old_pred_original_sample[-1] = None 

        return _latents, _old_pred_original_sample

    def shift_vip_image_rotary_grid_t(initial_grid_t, grid_t_queue):
        initial_grid_t[:-1] = np.copy(initial_grid_t[1:])
        initial_grid_t[-1] = grid_t_queue[0]
        
        grid_t_queue = grid_t_queue[1:]

        return initial_grid_t, grid_t_queue

#    def shift_image_embeddings(initial_embeddings, embeddings_queue):
#        initial_embeddings[:,:-1] = initial_embeddings[:,1:].clone()
#        initial_embeddings[:,-1] = embeddings_queue[:,0].clone()
#        embeddings_queue = embeddings_queue[:,1:]
#        return initial_embeddings, embeddings_queue
#
#    def shift_cond_frames(initial_frames, frames_queue):
#        initial_frames[:,:-1] = initial_frames[:,1:].clone()
#        initial_frames[:,-1] = frames_queue[:,0].clone()
#        frames_queue = frames_queue[:,1:]
#        return initial_frames, frames_queue

    # 8. fifo Denoising loop
    latents = prepare_fifo_latents()
    old_pred_original_sample = fifo_old_pred_original_sample
    for i in range(r_nf_per_chunk):
        old_pred_original_sample = [old_pred_original_sample[0],] + old_pred_original_sample
    if use_vip:
        if use_sliding_window_embedding:
            cond_frames, cond_frames_queue = prepare_fifo_cond_frames()
        #else:
        #    image_embeddings, image_embeddings_queue = prepare_fifo_image_embeddings()
        (
            vip_condition_rotary_grid_t,
            vip_condition_rotary_grid_h,
            vip_condition_rotary_grid_w
        ) = vip_condition_rotary_grid
        (
            vip_image_rotary_grid_t,
            vip_image_rotary_grid_h,
            vip_image_rotary_grid_w,
        ) = vip_image_rotary_grid
        (
            vip_image_rotary_grid_t, 
            vip_image_rotary_grid_t_queue
        ) = prepare_fifo_vip_image_rotary_grid_t(
            vip_image_rotary_grid_t
        )
        print("vip_image_rotary_grid_t", vip_image_rotary_grid_t)
        print("vip_image_rotary_grid_t_queue", vip_image_rotary_grid_t_queue)
        vip_condition_rotary_grid_t = prepare_fifo_vip_condition_rotary_grid_t(
            vip_condition_rotary_grid_t
        )
        print("vip_condition_rotary_grid_t", vip_condition_rotary_grid_t)
        image_embeddings = prepare_fifo_image_embeddings()

    if use_adaptive_padding and lookahead_denoising:
        queue_start_idx = num_inference_steps-l_nf_per_chunk
    else:
        queue_start_idx = 0

    fifo_video_latents = []

    if lookahead_denoising:
        fifo_timesteps = torch.cat([timesteps, torch.full((r_nf_per_chunk,), timesteps[-1], device=timesteps.device)])
        fifo_prev_timesteps = torch.cat([timesteps[1:], torch.full((r_nf_per_chunk+1,), -1, device=timesteps.device)])
        fifo_next_timesteps = torch.cat([torch.full((1,),-1,device=timesteps.device), timesteps[:-1], torch.full((r_nf_per_chunk,), timesteps[-2], device=timesteps.device)])
    else:
        fifo_timesteps = timesteps
        fifo_prev_timesteps = torch.cat([timesteps[1:], timesteps[-1:]])

    cache_latents = []
    for i in range(len(cache_idx)):
        cache_latents.append([])

    rank_list = list(reversed(range(2 * num_partitions if lookahead_denoising else num_partitions)))
    split = torch.linspace(0, len(rank_list), accelerator.num_processes+1).long()
    rank_list_per_gpu = rank_list[split[accelerator.process_index]:split[accelerator.process_index+1]]

    pbar = tqdm(
        total=num_frames + num_inference_steps - nf_per_chunk, 
        disable=not accelerator.is_local_main_process
    )
    for i in range(num_frames + num_inference_steps - nf_per_chunk):

        output_latents = latents.clone()
        output_old_pred_original_sample = copy.deepcopy(old_pred_original_sample)

        output_indexes = [[]]

        for rank in rank_list_per_gpu:
            start_idx = nf_per_chunk * (rank // 2) + r_nf_per_chunk * (rank%2) 
            prev_start_idx = nf_per_chunk * ((rank + 1)// 2) + r_nf_per_chunk * ((rank + 1)%2)

            if prev_start_idx <= queue_start_idx:
                output_indexes[0].append([0,0])
                break

            if rank % 2 == 1:
                midpoint_idx = start_idx + l_nf_per_chunk
            else:
                midpoint_idx = start_idx + r_nf_per_chunk
            real_end_idx = start_idx + nf_per_chunk
            if start_idx < queue_start_idx:
                start_idx = queue_start_idx
            end_idx = start_idx + nf_per_chunk

            t = fifo_timesteps.flip((0,))[start_idx:end_idx]
            prev_t = fifo_prev_timesteps.flip((0,))[start_idx:end_idx]
            next_t = fifo_next_timesteps.flip((0,))[start_idx:end_idx]
            input_latents = latents[:,start_idx:end_idx].clone()
            old_pred_sample = old_pred_original_sample[start_idx:end_idx]

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([input_latents] * 2) if do_classifier_free_guidance else input_latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            input_timesteps = t.expand(latent_model_input.shape[0],-1) 

            # obtain the video tokens
            if use_vip:
                # recompute rotary embeddings
                #print("vip_image_rotary_grid_t[start_idx:end_idx]", vip_image_rotary_grid_t[start_idx:end_idx])
                input_vip_image_rotary_emb = pipe._prepare_vip_rotary_positional_embeddings(
                    grid_t=vip_image_rotary_grid_t[start_idx:end_idx],
                    grid_h=vip_image_rotary_grid_h,
                    grid_w=vip_image_rotary_grid_w,
                    device=device,
                )
                vip_start_idx = find_embed_index(
                    vip_condition_rotary_grid_t,
                    vip_image_rotary_grid_t[start_idx]
                )
                #print("vip_condition_rotary_grid_t[vip_start_idx:(vip_start_idx+min(vip_nf_per_chunk+1, nf_per_chunk))]")
                #print(vip_condition_rotary_grid_t[vip_start_idx:(vip_start_idx+min(vip_nf_per_chunk+1, nf_per_chunk))])
                input_vip_condition_rotary_emb = pipe._prepare_vip_rotary_positional_embeddings(
                    grid_t=vip_condition_rotary_grid_t[vip_start_idx:(vip_start_idx+min(vip_nf_per_chunk+1, nf_per_chunk))],
                    grid_h=vip_condition_rotary_grid_h,
                    grid_w=vip_condition_rotary_grid_w,
                    device=device
                )

                if use_sliding_window_embedding:
                    input_image_embeddings = pipe._encode_image(
                        cond_frames[:,start_idx:end_idx].clone(),
                        latent_model_input.device,
                        do_classifier_free_guidance,
                        nf_per_chunk
                    )
                else:
                    input_image_embeddings = image_embeddings[:,vip_start_idx:(vip_start_idx+min(vip_nf_per_chunk+1, nf_per_chunk))].clone()

            # predict the noise residual
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=input_timesteps,
                image_rotary_emb=image_rotary_emb,
                vip_image_rotary_emb=input_vip_image_rotary_emb if use_vip else None,
                vip_condition_rotary_emb=input_vip_condition_rotary_emb if use_vip else None,
                vip_encoder_hidden_states=input_image_embeddings if use_vip else None,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
#            if i == 0:
#                if start_idx == 45 and end_idx == 58:
#                    print("latent_model_input", latent_model_input.mean(), latent_model_input.min(), latent_model_input.max())
#                    print("prompt_embeds", prompt_embeds.mean(), prompt_embeds.min(), prompt_embeds.max())
#                    print("input_timesteps", input_timesteps)
#                    print("image_rotary_emb", image_rotary_emb[0].mean(), image_rotary_emb[0].min(), image_rotary_emb[0].max())
#                    print("input_vip_image_rotary_emb", input_vip_image_rotary_emb[0].mean(), input_vip_image_rotary_emb[0].min(), input_vip_image_rotary_emb[0].max())
#                    print("input_vip_condition_rotary_emb", input_vip_condition_rotary_emb[0].mean(), input_vip_condition_rotary_emb[0].min(), input_vip_condition_rotary_emb[0].max())
#                    print("input_image_embeddings", input_image_embeddings.mean(), input_image_embeddings.min(), input_image_embeddings.max())
#                    print("noise_pred", noise_pred.mean(), noise_pred.min(), noise_pred.max())

            # perform guidance
            if use_dynamic_cfg:
                pipe._guidance_scale = 1 + guidance_scale * (
                    (1 - torch.cos(math.pi * ((num_inference_steps - t) / num_inference_steps) ** 5.0)) / 2
                )
                pipe._guidance_scale = pipe._guidance_scale[None,:,None,None,None]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1 
            assert isinstance(pipe.scheduler, CogVideoXDPMScheduler)
            _old_pred_sample = []
            for j in range(input_latents.shape[1]):
                _input_latents, x0 = pipe.scheduler.step(
                    noise_pred[:,[j]],
                    old_pred_sample[j],
                    t[j],
                    prev_t[j],
                    next_t[j] if next_t[j] > 0 else None,
                    input_latents[:,[j]],
                    **extra_step_kwargs,
                    return_dict=False
                )
#                if i == 0:
#                    if start_idx == 45 and end_idx == 58:
#                        print(j, "noise_pred[:,[j]]", noise_pred[:,[j]].mean(), noise_pred[:,[j]].min(), noise_pred[:,[j]].max())
#                        if old_pred_sample[j] is not None:
#                            print(j, "old_pred_sample[j]", old_pred_sample[j].mean(), old_pred_sample[j].min(), old_pred_sample[j].max())
#                        else:
#                            print(j, "old_pred_sample[j]", None)
#                        print(j, "input_latents[:,[j]]", input_latents[:,[j]].mean(), input_latents[:,[j]].min(), input_latents[:,[j]].max())
#                        print(j, "_input_latents", _input_latents.mean(), _input_latents.min(), _input_latents.max())
#                        print(j, "x0", x0.mean(), x0.min(), x0.max())
                input_latents[:,[j]] = _input_latents.to(prompt_embeds.dtype)
                _old_pred_sample.append( x0.to(prompt_embeds.dtype) )
                for _cn, _idx in enumerate(cache_idx):
                    if i <= _idx + num_inference_steps - nf_per_chunk and \
                       _idx + num_inference_steps - nf_per_chunk < i + num_inference_steps:
                        q_idx = _idx + num_inference_steps - nf_per_chunk - i + r_nf_per_chunk
                        if start_idx > queue_start_idx:
                            lb_idx = midpoint_idx
                            rb_idx = end_idx
                        elif start_idx == queue_start_idx:
                            lb_idx = max(r_nf_per_chunk,start_idx)
                            rb_idx = real_end_idx
                        else:
                            raise IOError
                        if lb_idx <= q_idx and q_idx < rb_idx:
                            l_idx = q_idx  - start_idx
                            if j == l_idx:
                                cache_latents[_cn].append(x0)

            if start_idx > queue_start_idx: 
                output_latents[:,midpoint_idx:end_idx] = input_latents[:,(midpoint_idx-start_idx):]
                output_old_pred_original_sample[midpoint_idx:end_idx] = _old_pred_sample[(midpoint_idx-start_idx):] 
                output_indexes[0].append([midpoint_idx, end_idx])
            elif start_idx == queue_start_idx:
                output_latents[:,max(r_nf_per_chunk,start_idx):real_end_idx] = input_latents[:,max(r_nf_per_chunk-start_idx, 0):(real_end_idx-start_idx)]
                output_old_pred_original_sample[max(r_nf_per_chunk,start_idx):real_end_idx] = _old_pred_sample[max(r_nf_per_chunk-start_idx, 0):(real_end_idx-start_idx)]
                output_indexes[0].append([max(r_nf_per_chunk,start_idx), real_end_idx])
            else:
                raise NotImplementedError
            del input_latents
            del _old_pred_sample

        # gather from all gpus
        print(accelerator.process_index, output_latents.mean())
        gathered_output_latents = accelerator.gather(output_latents.unsqueeze(0)) 
        print(gathered_output_latents.shape)
        print("gathered_output_latents", [item.mean() for item in gathered_output_latents]) 
        gathered_output_old_pred_original_sample = gather_object([output_old_pred_original_sample])
        gathered_output_indexes = gather_object(output_indexes)
        #print("gathered_output_indexes", gathered_output_indexes)
        assert len(gathered_output_latents) == len(gathered_output_old_pred_original_sample)
        assert len(gathered_output_old_pred_original_sample) == len(gathered_output_indexes)

        for gpu_id in range(gathered_output_latents.shape[0]):
            for index in gathered_output_indexes[gpu_id]:
                if index[0] >= index[1]:
                    continue
                output_latents[:,index[0]:index[1]] = gathered_output_latents[gpu_id,:,index[0]:index[1]]
                _tmp = []
                for item in gathered_output_old_pred_original_sample[gpu_id][index[0]:index[1]]:
                    if item is not None:
                        _tmp.append(item.to(device))
                    else:
                        _tmp.append(None)
                output_old_pred_original_sample[index[0]:index[1]] = _tmp
                del _tmp

        # reconstruct from latent to pixel space
        latents = output_latents
        old_pred_original_sample = output_old_pred_original_sample

        if accelerator.is_main_process:
            first_frame_idx = r_nf_per_chunk if lookahead_denoising else 0
            fifo_video_latents.append( latents[:,[first_frame_idx]] )
            latents, old_pred_original_sample = shift_latents(latents, old_pred_original_sample)

        # gather shifted latents from all gpus
        gathered_latents = accelerator.gather(latents.unsqueeze(0))
        gathered_old_pred_original_sample = gather_object([old_pred_original_sample])
        latents = gathered_latents[0]
        old_pred_original_sample = []
        for item in gathered_old_pred_original_sample[0]:
            if item is not None:
                old_pred_original_sample.append(item.to(device))
            else:
                old_pred_original_sample.append(None)

#        if i == 0:
#            print("old_pred_original_sample", [item.mean() for item in old_pred_original_sample[44:57]])

        if use_vip:
            if use_sliding_window_embedding:
                cond_frames, cond_frames_queue = shift_cond_frames(
                    cond_frames, cond_frames_queue
                )
            else:
                (
                    vip_image_rotary_grid_t, 
                    vip_image_rotary_grid_t_queue
                ) = shift_vip_image_rotary_grid_t(
                    vip_image_rotary_grid_t,
                    vip_image_rotary_grid_t_queue
                )
        queue_start_idx = max(0, queue_start_idx-1)
        pbar.update()

    if accelerator.is_main_process:
        latents = torch.cat(fifo_video_latents[(num_inference_steps - nf_per_chunk):], dim=1)
        latents = torch.cat(fifo_video_latents, dim=1)
    else:
        latents = None

    # gather cached latents
    gathered_cache_latents = gather_object([cache_latents])
    if accelerator.is_main_process:
        cache_latents = copy.deepcopy(gathered_cache_latents[0])
        for item in gathered_cache_latents[1:]:
            for cid in range(len(cache_latents)):
                cache_latents[cid] = cache_latents[cid] + [_item.to(device) for _item in item[cid]] 
        for cid in range(len(cache_latents)):
            print(cid, len(cache_latents[cid])) 

    return latents, cache_latents 

