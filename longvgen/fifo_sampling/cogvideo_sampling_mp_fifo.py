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
import torch.multiprocessing as mp

from diffusers.utils import BaseOutput

from longvgen.schedulers import CogVideoXDPMScheduler

@dataclass
class CogVideoXPipelineOutput(BaseOutput):

    frames: Union[List[np.ndarray], torch.FloatTensor]
    orig_frames: Union[List[np.ndarray], torch.FloatTensor]
    cache_frames: Union[List[np.ndarray], torch.FloatTensor]

@torch.no_grad()
def cogvideo_fifo_mp_v2(
    pipe_list,
    base_output,
    **kwargs
):
    # fetch parameters from base_output
    sampling_params = base_output.sampling_params
    use_sliding_window_embedding = sampling_params.get("use_sliding_window_embedding")
    lookahead_denoising = True
    use_adaptive_padding = sampling_params.get("use_adaptive_padding", True)
    num_partitions = sampling_params.get("num_partitions", 4)

    fifo_latents = base_output.fifo_latents
    fifo_old_pred_original_sample = base_output.fifo_old_pred_original_sample
    nf_per_chunk = base_output.nf_per_chunk
    vip_nf_per_chunk = base_output.vip_nf_per_chunk
    num_frames = base_output.num_frames
    image_embeddings = base_output.image_embeddings
    timesteps = base_output.timesteps
    num_inference_steps = base_output.num_inference_steps
    do_classifier_free_guidance = base_output.do_classifier_free_guidance
    use_separate_guidance = base_output.use_separate_guidance
    use_dynamic_cfg = base_output.use_dynamic_cfg
    prompt_embeds = base_output.prompt_embeds
    image_rotary_emb = base_output.image_rotary_emb
    vip_image_rotary_grid = base_output.vip_image_rotary_grid
    vip_condition_rotary_grid = base_output.vip_condition_rotary_grid
    attention_kwargs = base_output.attention_kwargs
    guidance_scale = base_output.guidance_scale
    guidance_scale_img = base_output.guidance_scale_img
    extra_step_kwargs = base_output.extra_step_kwargs
    cache_idx = base_output.cache_idx
    condition_frames = base_output.condition_frames
    video_ipadapter_start_frame_idx = base_output.video_ipadapter_start_frame_idx
    output_type = base_output.output_type
    return_dict = base_output.return_dict
    orig_latents = base_output.orig_latents

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

    def shift_latents(_latents, _old_pred_original_sample):
        # shift latents
        _latents[:,:-1] = _latents[:,1:].clone()
        _old_pred_original_sample[:-1] = copy.deepcopy(_old_pred_original_sample[1:])

        # add new noise to the last frame
        #_latents[:,-1] = torch.randn_like(_latents[:,-1])
        _latents[:,-1] = pipe_list[0].scheduler.add_noise_to_xt(
            _latents[:,-1],
            torch.randn_like(_latents[:,-1]),
            torch.Tensor([999]).long()
        )
        _old_pred_original_sample[-1] = None 

        return _latents, _old_pred_original_sample

    def shift_vip_image_rotary_grid_t(initial_grid_t, grid_t_queue):
        initial_grid_t[:-1] = np.copy(initial_grid_t[1:])
        initial_grid_t[-1] = grid_t_queue[0]
        
        grid_t_queue = grid_t_queue[1:]

        return initial_grid_t, grid_t_queue

    # prepare the tensors necessary for the queue
    latents = prepare_fifo_latents()
    old_pred_original_sample = fifo_old_pred_original_sample

    for i in range(r_nf_per_chunk):
        old_pred_original_sample = [old_pred_original_sample[0],] + old_pred_original_sample

    if use_vip:
        if use_sliding_window_embedding:
            cond_frames, cond_frames_queue = prepare_fifo_cond_frames()
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

    # prepare the multi-processing queue 
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    processes = [] 

    num_processes = len(pipe_list)
    for pid in range(num_processes):
        p = mp.Process(
            target = fifo_onestep_per_gpu,
            args = (
                pid,
                input_queue,
                output_queue,
                pipe_list[pid],
                prompt_embeds.to(pipe_list[pid].device),
                tuple([item.to(pipe_list[pid].device) for item in image_rotary_emb]),
                num_inference_steps,
                do_classifier_free_guidance,
                use_separate_guidance,
                guidance_scale,
                guidance_scale_img,
                use_dynamic_cfg,
                attention_kwargs,
                #extra_step_kwargs
            )
        )
        p.start()
        processes.append(p)
    
    num_rank = 2 * num_partitions if lookahead_denoising else num_partitions
    num_base_rank = num_rank // num_processes if num_rank % num_processes == 0 else num_rank // num_processes + 1

    # begin fifo sampling loop
    pbar = tqdm(
        total=num_frames + num_inference_steps - nf_per_chunk, 
    )
    for i in range(num_frames + num_inference_steps - nf_per_chunk):

        output_latents = latents.clone()
        output_old_pred_original_sample = copy.deepcopy(old_pred_original_sample)

        for base_rank in range(num_base_rank):
            input_processes = 0
            for sub_rank in range(num_processes):
                rank = base_rank * num_processes + sub_rank
                if rank < num_rank:
                    start_idx = nf_per_chunk * (rank // 2) + r_nf_per_chunk * (rank%2) 
                    prev_start_idx = nf_per_chunk * ((rank + 1)// 2) + r_nf_per_chunk * ((rank + 1)%2)

                    if prev_start_idx <= queue_start_idx:
                        continue

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

                    if use_vip:
                        input_vip_image_rotary_grid_t = vip_image_rotary_grid_t[start_idx:end_idx].copy()
                        input_vip_image_rotary_grid_h = vip_image_rotary_grid_h.copy()
                        input_vip_image_rotary_grid_w = vip_image_rotary_grid_w.copy()
                        vip_start_idx = find_embed_index(
                            vip_condition_rotary_grid_t,
                            vip_image_rotary_grid_t[start_idx]
                        )
                        input_vip_condition_rotary_grid_t = vip_condition_rotary_grid_t[vip_start_idx:(vip_start_idx+min(vip_nf_per_chunk+1, nf_per_chunk))].copy()
                        input_vip_condition_rotary_grid_h = vip_condition_rotary_grid_h.copy()
                        input_vip_condition_rotary_grid_w = vip_condition_rotary_grid_w.copy()
                        
                        input_image_embeddings = image_embeddings[:,vip_start_idx:(vip_start_idx+min(vip_nf_per_chunk+1, nf_per_chunk))].clone()

                    else:
                        input_vip_image_rotary_grid_t = None
                        input_vip_image_rotary_grid_h = None
                        input_vip_image_rotary_grid_w = None
                        input_vip_condition_rotary_grid_t = None
                        input_vip_condition_rotary_grid_h = None
                        input_vip_condition_rotary_grid_w = None
                        input_image_embeddings = None

                    input_queue.put((
                        i,
                        sub_rank,
                        queue_start_idx,
                        start_idx,
                        midpoint_idx,
                        end_idx,
                        real_end_idx,
                        t,
                        prev_t,
                        next_t,
                        input_latents,
                        old_pred_sample,
                        input_vip_image_rotary_grid_t,
                        input_vip_image_rotary_grid_h,
                        input_vip_image_rotary_grid_w,
                        input_vip_condition_rotary_grid_t,
                        input_vip_condition_rotary_grid_h,
                        input_vip_condition_rotary_grid_w,
                        input_image_embeddings,
                        cache_idx
                    ))
                    input_processes += 1

            while input_processes > 0:
                outputs = output_queue.get()
                if outputs is None:
                    continue
                (
                    sub_rank,
                    start_idx,
                    midpoint_idx,
                    end_idx,
                    real_end_idx,
                    output_latents_per_gpu,
                    output_old_pred_sample_per_gpu,
                    cache_latents_per_gpu,
                ) = outputs
                if start_idx > queue_start_idx: 
                    output_latents[:,midpoint_idx:end_idx] = output_latents_per_gpu[:,(midpoint_idx-start_idx):]
                    output_old_pred_original_sample[midpoint_idx:end_idx] = output_old_pred_sample_per_gpu[(midpoint_idx-start_idx):] 
                elif start_idx == queue_start_idx:
                    output_latents[:,max(r_nf_per_chunk,start_idx):real_end_idx] = output_latents_per_gpu[:,max(r_nf_per_chunk-start_idx, 0):(real_end_idx-start_idx)]
                    output_old_pred_original_sample[max(r_nf_per_chunk,start_idx):real_end_idx] = output_old_pred_sample_per_gpu[max(r_nf_per_chunk-start_idx, 0):(real_end_idx-start_idx)]
                else:
                    raise NotImplementedError
                for cid in cache_latents:
                    cache_latents[cid] = cache_latents[cid] + cache_latents_per_gpu[cid] 
                del output_latents_per_gpu
                del output_old_pred_sample_per_gpu
                input_processes -= 1

        # reconstruct from latent to pixel space
        latents = output_latents
        old_pred_original_sample = output_old_pred_original_sample

        first_frame_idx = r_nf_per_chunk if lookahead_denoising else 0
        fifo_video_latents.append( latents[:,[first_frame_idx]] )
        latents, old_pred_original_sample = shift_latents(latents, old_pred_original_sample)


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

    for _ in range(num_processes):
        input_queue.put(None)
    for p in processes:
        p.join()
        p.close()

    latents = torch.cat(fifo_video_latents[(num_inference_steps - nf_per_chunk):], dim=1)
    #latents = torch.cat(fifo_video_latents, dim=1)
    for i in range(len(cache_latents)):
        cache_latents[i] = torch.cat(cache_latents[i], dim=1)

    if not output_type == "latent":
        video = pipe_list[0].decode_latents(latents)
        video = pipe_list[0].video_processor.postprocess_video(video=video, output_type=output_type)

        orig_video = pipe_list[0].decode_latents(orig_latents)
        orig_video = pipe_list[0].video_processor.postprocess_video(video=orig_video, output_type=output_type)
        cache_video = []
        for i in range(len(cache_latents)):
            cache_video.append(
                pipe_list[0].video_processor.postprocess_video(
                    video = pipe_list[0].decode_latents(cache_latents[i]),
                    output_type=output_type
                )
            )

    else:
        video = latents
        orig_video = orig_latents
        cache_video = cache_latents

    if not return_dict:
        return (orig_video, video, cache_video)

    return CogVideoXPipelineOutput(frames=video, orig_frames=orig_video, cache_frames=cache_video)

def move_list_to_device(x, device):
    assert isinstance(x, list)
    new_x = []
    for item in x:
        if item is None:
            new_x.append(None)
        else:
            new_x.append(item.to(device))
    return new_x

@torch.no_grad()
def fifo_onestep_per_gpu(
    pid,
    input_queue,
    output_queue,
    pipe,
    prompt_embeds,
    image_rotary_emb,
    num_inference_steps,
    do_classifier_free_guidance,
    use_separate_guidance,
    guidance_scale,
    guidance_scale_img,
    use_dynamic_cfg,
    attention_kwargs,
    #extra_step_kwargs
):
    print(f"{pid}th process started")

    while True:
        inputs = input_queue.get()
        if inputs is None:
            print("process " + str(pid) + " is done") 
            return

        (
            i,
            sub_rank,
            queue_start_idx,
            start_idx,
            midpoint_idx,
            end_idx,
            real_end_idx,
            t,
            prev_t,
            next_t,
            input_latents,
            old_pred_sample,
            input_vip_image_rotary_grid_t,
            input_vip_image_rotary_grid_h,
            input_vip_image_rotary_grid_w,
            input_vip_condition_rotary_grid_t,
            input_vip_condition_rotary_grid_h,
            input_vip_condition_rotary_grid_w,
            input_image_embeddings,
            cache_idx
        ) = inputs
        
        cache_latents = []
        for i in range(len(cache_idx)):
            cache_latents.append([])
        
        nf_per_chunk = input_latents.shape[1]
        l_nf_per_chunk = nf_per_chunk-nf_per_chunk//2
        r_nf_per_chunk = nf_per_chunk // 2

        # move all the data to the device
        device = prompt_embeds.device
        t = t.to(device)
        prev_t = prev_t.to(device)
        next_t = next_t.to(device)
        input_latents = input_latents.to(device)
        output_latents = input_latents.clone()
        old_pred_sample = move_list_to_device(old_pred_sample, device)

        use_vip = (input_image_embeddings is not None)
        if use_vip:
            input_image_embeddings = input_image_embeddings.to(device)

            # obtain the video tokens
            # recompute rotary embeddings
            input_vip_image_rotary_emb = pipe._prepare_vip_rotary_positional_embeddings(
                grid_t=input_vip_image_rotary_grid_t,
                grid_h=input_vip_image_rotary_grid_h,
                grid_w=input_vip_image_rotary_grid_w,
                device=device,
            )
            input_vip_condition_rotary_emb = pipe._prepare_vip_rotary_positional_embeddings(
                grid_t=input_vip_condition_rotary_grid_t,
                grid_h=input_vip_condition_rotary_grid_h,
                grid_w=input_vip_condition_rotary_grid_w,
                device=device
            )

        # expand the latents if we are doing classifier free guidance
        if do_classifier_free_guidance:
            if use_separate_guidance:
                latent_model_input = torch.cat([input_latents] * 3)
            else:
                latent_model_input = torch.cat([input_latents] * 2)
        else:
            latent_model_input = input_latents

        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        input_timesteps = t.expand(latent_model_input.shape[0],-1) 

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

        # perform guidance
        if use_dynamic_cfg:
            pipe._guidance_scale = 1 + guidance_scale * (
                (1 - torch.cos(math.pi * ((num_inference_steps - t) / num_inference_steps) ** 5.0)) / 2
            )
            pipe._guidance_scale = pipe._guidance_scale[None,:,None,None,None]
            guidance_scale_img = 1 + guidance_scale_img * (
                (1 - torch.cos(math.pi * ((num_inference_steps - t) / num_inference_steps) ** 5.0)) / 2
            )
            guidance_scale_img = guidance_scale_img[None,:,None,None,None]
        if do_classifier_free_guidance:
            if use_separate_guidance:
                noise_pred_uncond_txt, noise_pred_uncond_img, noise_pred_txt_img = noise_pred.chunk(3)
                noise_pred = noise_pred_txt_img + (pipe.guidance_scale-1) * (noise_pred_txt_img - noise_pred_uncond_txt) + (guidance_scale_img - 1) * (noise_pred_txt_img - noise_pred_uncond_img)
            else:
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
                #**pipe.extra_step_kwargs,
                return_dict=False
            )
            output_latents[:,[j]] = _input_latents.to(prompt_embeds.dtype)
            _old_pred_sample.append( x0.to(prompt_embeds.dtype) )
            
            # save cached latents
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

        output_queue.put((
            sub_rank,
            start_idx,
            midpoint_idx,
            end_idx,
            real_end_idx,
            output_latents,
            _old_pred_sample,
            cache_latents,
        ))
        del output_latents
        del _old_pred_sample
            
