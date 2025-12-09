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

from longvgen.schedulers import CogVideoXDPMScheduler

def cogvideo_fifo(
    pipe,
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
        #print("new shifting methods")
        #_latents[:,-1] = pipe.scheduler.add_noise_to_xt(
        #    _latents[:,-1],
        #    torch.randn_like(_latents[:,-1]),
        #    torch.Tensor([999]).long()
        #)
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

    pbar = tqdm(total=num_frames + num_inference_steps - nf_per_chunk)
    for i in range(num_frames + num_inference_steps - nf_per_chunk):
        output_latents = latents.clone()
        output_old_pred_original_sample = copy.deepcopy(old_pred_original_sample)

        for rank in reversed(range(2 * num_partitions if lookahead_denoising else num_partitions)):
            
            pbar.set_description(f"Processing frame {i} rank {rank}/{2*num_partitions}")

            start_idx = nf_per_chunk * (rank // 2) + r_nf_per_chunk * (rank%2) 
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

            #print(queue_start_idx, start_idx, midpoint_idx, end_idx, real_end_idx)
            #print("latent_model_input", latent_model_input.shape)
            #print("image_rotary_emb", [item.shape for item in image_rotary_emb])
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
            if i == 0:
                if start_idx == 45 and end_idx == 58:
                    print("latent_model_input", latent_model_input.mean(), latent_model_input.min(), latent_model_input.max())
                    print("prompt_embeds", prompt_embeds.mean(), prompt_embeds.min(), prompt_embeds.max())
                    print("input_timesteps", input_timesteps)
                    print("image_rotary_emb", image_rotary_emb[0].mean(), image_rotary_emb[0].min(), image_rotary_emb[0].max())
                    print("input_vip_image_rotary_emb", input_vip_image_rotary_emb[0].mean(), input_vip_image_rotary_emb[0].min(), input_vip_image_rotary_emb[0].max())
                    print("input_vip_condition_rotary_emb", input_vip_condition_rotary_emb[0].mean(), input_vip_condition_rotary_emb[0].min(), input_vip_condition_rotary_emb[0].max())
                    print("input_image_embeddings", input_image_embeddings.mean(), input_image_embeddings.min(), input_image_embeddings.max())
                    print("noise_pred", noise_pred.mean(), noise_pred.min(), noise_pred.max())
            #print("noise_pred", noise_pred.shape)
            #print("t", t)
            #print("prev_t", prev_t)
            #print("next_t", next_t)

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
                if i == 0:
                    if start_idx == 45 and end_idx == 58:
                        print(j, "noise_pred[:,[j]]", noise_pred[:,[j]].mean(), noise_pred[:,[j]].min(), noise_pred[:,[j]].max())
                        if old_pred_sample[j] is not None:
                            print(j, "old_pred_sample[j]", old_pred_sample[j].mean(), old_pred_sample[j].min(), old_pred_sample[j].max())
                        else:
                            print(j, "old_pred_sample[j]", None)
                        print(j, "input_latents[:,[j]]", input_latents[:,[j]].mean(), input_latents[:,[j]].min(), input_latents[:,[j]].max())
                        print(j, "_input_latents", _input_latents.mean(), _input_latents.min(), _input_latents.max())
                        print(j, "x0", x0.mean(), x0.min(), x0.max())
                input_latents[:,[j]] = _input_latents.to(prompt_embeds.dtype)
                _old_pred_sample.append( x0.to(prompt_embeds.dtype) )
                for _cn, _idx in enumerate(cache_idx):
                    if i <= _idx + num_inference_steps - nf_per_chunk and \
                       _idx + num_inference_steps - nf_per_chunk < i + num_inference_steps:
                        q_idx = _idx + num_inference_steps - nf_per_chunk - i + r_nf_per_chunk
                        if start_idx > queue_start_idx:
                            lb_idx = midpoint_idx
                        elif start_idx == queue_start_idx:
                            lb_idx = max(r_nf_per_chunk,start_idx)
                        else:
                            raise IOError
                        if lb_idx <= q_idx and q_idx < end_idx:
                            l_idx = q_idx  - start_idx
                            if j == l_idx:
                                cache_latents[_cn].append(x0.unsqueeze(2))

            if lookahead_denoising:
                if start_idx > queue_start_idx: 
                    output_latents[:,midpoint_idx:end_idx] = input_latents[:,(midpoint_idx-start_idx):]
                    output_old_pred_original_sample[midpoint_idx:end_idx] = _old_pred_sample[(midpoint_idx-start_idx):] 
                elif start_idx == queue_start_idx:
                    #tmp = input_latents
                    #print("input_latents", [tmp[:,i].mean() for i in range(tmp.shape[1])])
                    #tmp = _old_pred_sample
                    #print("original", [tmp[i].mean() for i in range(len(tmp))])

                    output_latents[:,max(r_nf_per_chunk,start_idx):real_end_idx] = input_latents[:,max(r_nf_per_chunk-start_idx, 0):(real_end_idx-start_idx)]
                    output_old_pred_original_sample[max(r_nf_per_chunk,start_idx):real_end_idx] = _old_pred_sample[max(r_nf_per_chunk-start_idx, 0):(real_end_idx-start_idx)]
                    break
                else:
                    raise NotImplementedError
            else:
                latents[:,start_idx:end_idx] = input_latents
            del input_latents
            del _old_pred_sample


        # reconstruct from latent to pixel space
        latents = output_latents
        old_pred_original_sample = output_old_pred_original_sample

        first_frame_idx = r_nf_per_chunk if lookahead_denoising else 0
        fifo_video_latents.append( latents[:,[first_frame_idx]] )
        latents, old_pred_original_sample = shift_latents(latents, old_pred_original_sample)
        if i == 0:
            print("old_pred_original_sample", [item.mean() for item in old_pred_original_sample[44:57]])
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

    latents = torch.cat(fifo_video_latents[(num_inference_steps - nf_per_chunk):], dim=1)
    #latents = torch.cat(fifo_video_latents, dim=1)

    return latents, cache_latents 

#def fifo_freeinit(
#    pipe,
#    latents,
#    nf_per_chunk,
#    num_frames,
#    image_embeddings,
#    timesteps,
#    num_inference_steps,
#    do_classifier_free_guidance,
#    prompt_embeds,
#    cross_attention_kwargs,
#    guidance_scale,
#    extra_step_kwargs,
#    cache_idx: Optional[List[int]] = [],
#    condition_frames: Optional[torch.Tensor] = None,
#    lookahead_denoising: bool=True,
#    num_partitions: int=4,
#    use_adaptive_padding: bool=False,
#    use_sliding_window_embedding: bool=False,
#    **kwargs
#):
#    assert lookahead_denoising
#    if use_sliding_window_embedding:
#        assert condition_frames is not None
#
#    def prepare_fifo_latents():
#        latents_list = []
#        if lookahead_denoising:
#            for i in range(nf_per_chunk//2):
#                latents_list.append( 
#                    pipe.scheduler.add_noise(
#                        latents[:,:,[0]], torch.randn_like(latents[:,:,[0]]), 
#                        timesteps[-1:]
#                    )
#                )
#
#        for i, t in enumerate(timesteps.flip((0,))):
#            latents_list.append(
#                pipe.scheduler.add_noise(
#                    latents[:,:,[max(0, i-(num_inference_steps-nf_per_chunk))]],
#                    torch.randn_like(latents[:,:,[0]]),
#                    torch.Tensor([t]).long().to(latents.device)
#                )
#            )
#
#        latents_queue = pipe.scheduler.add_noise(
#            latents[:,:,nf_per_chunk:],
#            torch.randn_like(latents[:,:,nf_per_chunk:]),
#            timesteps[0:1]
#        )
#
#        return torch.cat(latents_list, dim=2), latents_queue
#
#    def prepare_fifo_image_embeddings(): 
#        
#        embeddings_list = []
#        if lookahead_denoising: 
#            embeddings_list += [image_embeddings[:,[0]],] * (nf_per_chunk//2)
#        
#        embeddings_list += [image_embeddings[:,[0]],] * (num_inference_steps-nf_per_chunk)
#        embeddings_list += [image_embeddings[:,:nf_per_chunk],]
#
#        initial_embeddings = torch.cat(embeddings_list, dim=1)
#
#        embeddings_queue = torch.cat(
#            [image_embeddings[:,nf_per_chunk:],]+ [image_embeddings[:,[-1]],] * num_inference_steps,
#            dim = 1
#        )
#
#        return initial_embeddings, embeddings_queue            
#            
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
#
#    def shift_latents(_latents, _latents_queue):
#        # shift latents
#        _latents[:,:,:-1] = _latents[:,:,1:].clone()
#
#        # add new noise to the last frame
#        if _latents_queue.shape[2] > 0:
#            _latents[:,:,-1] = _latents_queue[:,:,0]
#            _latents_queue = _latents_queue[:,:,1:]
#        else:
#            _latents[:,:,-1] = torch.randn_like(_latents[:,:,-1])
#            
#        # shift queue
#
#        return _latents, _latents_queue
#
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
#
#    # 8. fifo Denoising loop
#    latents, latents_queue = prepare_fifo_latents()
#    if use_sliding_window_embedding:
#        cond_frames, cond_frames_queue = prepare_fifo_cond_frames()
#    else:
#        image_embeddings, image_embeddings_queue = prepare_fifo_image_embeddings()
#
#    if use_adaptive_padding and lookahead_denoising:
#        queue_start_idx = num_inference_steps-nf_per_chunk + nf_per_chunk//2
#    else:
#        queue_start_idx = 0
#
#    fifo_video_latents = []
#
#    if lookahead_denoising:
#        fifo_timesteps = torch.cat([timesteps, torch.full((nf_per_chunk//2,), timesteps[-1], device=timesteps.device)])
#        fifo_prev_timesteps = torch.cat([timesteps[1:], torch.full((nf_per_chunk//2+1,), timesteps[-1], device=timesteps.device)])
#    else:
#        fifo_timesteps = timesteps
#        fifo_prev_timesteps = torch.cat([timesteps[1:], timesteps[-1:]])
#
#    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
#    cache_latents = []
#    for i in range(len(cache_idx)):
#        cache_latents.append([])
#
#    for i in trange(num_frames + num_inference_steps - nf_per_chunk, desc="fifo sampling"):
#        for rank in reversed(range(2 * num_partitions if lookahead_denoising else num_partitions)):
#            start_idx = rank*(nf_per_chunk // 2) if lookahead_denoising else rank*nf_per_chunk
#            if start_idx < queue_start_idx:
#                start_idx = queue_start_idx
#            midpoint_idx = start_idx + nf_per_chunk // 2
#            end_idx = start_idx + nf_per_chunk
#
#            t = fifo_timesteps.flip((0,))[start_idx:end_idx]
#            prev_t = fifo_prev_timesteps.flip((0,))[start_idx:end_idx]
#            input_latents = latents[:,:,start_idx:end_idx].clone()
#
#            # expand the latents if we are doing classifier free guidance
#            latent_model_input = torch.cat([input_latents] * 2) if do_classifier_free_guidance else input_latents
#            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
#
#            # obtain the video tokens
#            if use_sliding_window_embedding:
#                input_image_embeddings = pipe._encode_image(
#                    cond_frames[:,start_idx:end_idx].clone(),
#                    latent_model_input.device,
#                    do_classifier_free_guidance,
#                    nf_per_chunk
#                )
#            else:
#                input_image_embeddings = image_embeddings[:,start_idx:end_idx].clone()
#
#            # predict the noise residual
#            noise_pred = pipe.unet(
#                latent_model_input,
#                torch.stack([t,] * 2) if do_classifier_free_guidance else t,
#                context=prompt_embeds,
#                ext_context=input_image_embeddings,
#                cross_attention_kwargs=cross_attention_kwargs,
#                return_dict=False,
#            )[0]
#
#            # perform guidance
#            if do_classifier_free_guidance:
#                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#
#            # reshape latents
#            bsz, channel, frames, width, height = input_latents.shape
#            #input_latents = input_latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
#            #noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
#
#            # compute the previous noisy sample x_t -> x_t-1
#            for j in range(input_latents.shape[2]):
#                input_latents[:,:,j], x0 = pipe.scheduler.step(
#                    noise_pred[:,:,j],
#                    t[j],
#                    prev_t[j],
#                    input_latents[:,:,j],
#                    return_dict=False,
#                    **extra_step_kwargs
#                )
#                for _cn, _idx in enumerate(cache_idx):
#                    if i <= _idx + num_inference_steps - nf_per_chunk and \
#                       _idx + num_inference_steps - nf_per_chunk < i + num_inference_steps:
#                        q_idx = _idx + num_inference_steps - nf_per_chunk - i + nf_per_chunk//2
#                        if start_idx > queue_start_idx:
#                            lb_idx = midpoint_idx
#                        elif start_idx == queue_start_idx:
#                            lb_idx = max(nf_per_chunk//2,start_idx)
#                        else:
#                            raise IOError
#                        if lb_idx <= q_idx and q_idx < end_idx:
#                            l_idx = q_idx  - start_idx
#                            if j == l_idx:
#                                cache_latents[_cn].append(x0.unsqueeze(2))
#
#            # reshape latents back
#            #latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)
#
#            if lookahead_denoising:
#                if start_idx > queue_start_idx: 
#                    latents[:,:,midpoint_idx:end_idx] = input_latents[:,:,-(nf_per_chunk//2):]
#                elif start_idx == queue_start_idx:
#                    latents[:,:,max(nf_per_chunk//2,start_idx):end_idx] = input_latents[:,:,max(nf_per_chunk//2-start_idx, 0):]
#                    break
#                else:
#                    raise NotImplementedError
#            else:
#                latents[:,:,start_idx:end_idx] = input_latents
#            del input_latents
#
#        # reconstruct from latent to pixel space
#        first_frame_idx = nf_per_chunk // 2 if lookahead_denoising else 0
#        fifo_video_latents.append( latents[:,:,[first_frame_idx]] )
#        latents, latents_queue = shift_latents(latents, latents_queue)
#        if use_sliding_window_embedding:
#            cond_frames, cond_frames_queue = shift_cond_frames(
#                cond_frames, cond_frames_queue
#            )
#        else:
#            image_embeddings, image_embeddings_queue = shift_image_embeddings(
#                image_embeddings, image_embeddings_queue
#            )
#        queue_start_idx = max(0, queue_start_idx-1)
#
#    latents = torch.cat(fifo_video_latents[(num_inference_steps - nf_per_chunk):], dim=2)
#
#    return latents, cache_latents 
#
#def denoising_together(
#    pipe,
#    latents,
#    nf_per_chunk,
#    num_frames,
#    image_embeddings,
#    timesteps,
#    num_inference_steps,
#    do_classifier_free_guidance,
#    prompt_embeds,
#    cross_attention_kwargs,
#    guidance_scale,
#    extra_step_kwargs,
#    cache_idx: Optional[List[int]] = [],
#    num_partitions: int=4,
#    **kwargs
#):
#    def diffusion_onestep(latents, t, prev_t):
#        # expand the latents if we are doing classifier free guidance
#        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
#        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
#
#        # predict the noise residual
#        noise_pred = pipe.unet(
#            latent_model_input,
#            t,
#            context=prompt_embeds,
#            ext_context=image_embeddings[:,:nf_per_chunk],
#            cross_attention_kwargs=cross_attention_kwargs,
#            return_dict=False,
#        )[0]
#
#        # perform guidance
#        if do_classifier_free_guidance:
#            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#
#        # reshape latents
#        bsz, channel, frames, width, height = latents.shape
#        latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
#        noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
#
#        # compute the previous noisy sample x_t -> x_t-1
#        #prev_t = timesteps[i+1] if i+1 < len(timesteps) else timesteps[-1]
#        latents = pipe.scheduler.step(noise_pred, t, prev_t, latents, **extra_step_kwargs).prev_sample
#
#        # reshape latents back
#        latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)
#
#        return latents
#
#    # 7. fifo Denoising loop
#    pre_1st_latents = latents[:,:,:nf_per_chunk].clone()
#    pre_latents = None
#
#
#    fifo_timesteps = timesteps
#    fifo_prev_timesteps = torch.cat([timesteps[1:], timesteps[-1:]])
#
#    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
#    cache_latents = []
#    for i in range(len(cache_idx)):
#        cache_latents.append([])
#
#    for rank in reversed(range(num_partitions)):
#        start_idx = rank*nf_per_chunk
#        end_idx = start_idx + nf_per_chunk
#
#        t = fifo_timesteps.flip((0,))[start_idx:end_idx]
#        prev_t = fifo_prev_timesteps.flip((0,))[start_idx:end_idx]
#        # denoise process for 1st latents 
#        latents = []
#        for _i in trange(0, nf_per_chunk, 1, desc=f"step {rank}: fifo sampling for 1st chunk"):
#            latents = [pre_1st_latents[:,:,[-_i-1]],] + latents
#            pre_1st_latents = diffusion_onestep(
#                pre_1st_latents,
#                t[-_i-1],
#                prev_t[-_i-1]
#            )
#
#        latents = torch.cat(latents, dim=2)
#    
#        fifo_video_latents = []
#        image_embeddings_queue = torch.cat([image_embeddings] + [image_embeddings[:,[-1]]] * (rank+1) * nf_per_chunk, dim=1)
#
#        for i in trange(num_frames + rank * nf_per_chunk, desc=f"rank {rank}: fifo sampling"):
#
#            # expand the latents if we are doing classifier free guidance
#            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else input_latents
#            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
#
#            # obtain the video tokens
#            input_image_embeddings = image_embeddings_queue[:,i:i+nf_per_chunk].clone()
#
#            # predict the noise residual
#            noise_pred = pipe.unet(
#                latent_model_input,
#                torch.stack([t,] * 2) if do_classifier_free_guidance else t,
#                context=prompt_embeds,
#                ext_context=input_image_embeddings,
#                cross_attention_kwargs=cross_attention_kwargs,
#                return_dict=False,
#            )[0]
#
#            # perform guidance
#            if do_classifier_free_guidance:
#                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#
#            # reshape latents
#            bsz, channel, frames, width, height = latents.shape
#            #input_latents = input_latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
#            #noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
#
#            # compute the previous noisy sample x_t -> x_t-1
#            for j in range(latents.shape[2]):
#                latents[:,:,j], x0 = pipe.scheduler.step(
#                    noise_pred[:,:,j],
#                    t[j],
#                    prev_t[j],
#                    latents[:,:,j],
#                    return_dict=False,
#                    **extra_step_kwargs
#                )
#                for _cn, _idx in enumerate(cache_idx):
#                    if i <= _idx and _idx < i + nf_per_chunk:
#                        # cache_idx is in the queue
#                        q_idx = _idx - i
#                        if j == q_idx:
#                            #print(q_idx)
#                            cache_latents[_cn].append(x0.unsqueeze(2))
#                            #print(_cn, q_idx, len(cache_latents[_cn]))
#
#            # reshape latents back
#            #latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)
#
#            fifo_video_latents.append(latents[:,:,[0]])
#            # shift latents
#            latents[:,:,:-1] = latents[:,:,1:].clone()
#            # add new noise to the last frame
#            if pre_latents is None:
#                latents[:,:,-1] = torch.randn_like(latents[:,:,-1])
#            else:
#                latents[:,:,-1] = pre_latents[:,:,0].clone()
#                pre_latents = pre_latents[:,:,1:] if pre_latents.shape[2] > 1 else pre_latents
#
#
#        fifo_video_latents = torch.cat(fifo_video_latents, dim=2)
#        pre_latents = fifo_video_latents[:,:,nf_per_chunk:]
#        pre_1st_latents = fifo_video_latents[:,:,:nf_per_chunk]
#
#    return fifo_video_latents, cache_latents
