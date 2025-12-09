"""
This script demonstrates how to generate a video from a text prompt using CogVideoX with 🤗Huggingface Diffusers Pipeline.

Note:
    This script requires the `diffusers>=0.30.0` library to be installed， after `diffusers 0.31.0` release,
    need to update.

Run the script:
    $ python cli_demo.py --prompt "A girl ridding a bike." --model_path THUDM/CogVideoX-2b

"""
import inspect
from omegaconf import OmegaConf
import argparse
import os
import datetime
import copy
import json
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns
#from sklearn import datasets
from sklearn import manifold
from einops import rearrange
from pathlib import Path
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    gather_object,
    DistributedDataParallelKwargs, 
    ProjectConfiguration, 
    set_seed,
    InitProcessGroupKwargs
)
from datetime import timedelta
import torch.multiprocessing as mp
import threading as th
from queue import Queue


import torch

from diffusers import (
    CogVideoXPipeline,
    CogVideoXDDIMScheduler, 
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.schedulers import CogVideoXDPMScheduler as OrigCogVideoXDPMScheduler
from diffusers.utils import (
    export_to_video, 
    load_image,
    check_min_version
)
from transformers import AutoImageProcessor, AutoModel

from huggingface_hub import hf_hub_download
from huggingface_hub import HfFileSystem

from longvgen.models import CogVideoXTransformer3DModel 
from longvgen.pipeline import (
    MPFIFOVideoIPAdapterCogVideoXPipeline,
    LongVGenCogVideoXPipeline, 
)
from longvgen.schedulers import CogVideoXDPMScheduler
from longvgen.data.long_video import load_video
from longvgen.video_ipadapter import Resampler
from longvgen.fifo_sampling import cogvideo_fifo_mp_v2

def create_output_folders(output_dir, config, prefix="longvgen"):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"{prefix}_{now}")
    os.makedirs(out_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))
    return out_dir

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

def init_pipeline_mp(gpu_rank, args, dtype, input_queue, output_queue):
    gpu_id = input_queue.get()
    if gpu_id is None:
        print("init pipeline on " + str(gpu_rank) + " is done") 
        return

    use_vip = args.use_vip
    if use_vip:
        vip_params = args.video_ipadapter_params
        use_vae_as_encoder = vip_params.use_vae_as_encoder
        vip_path = args.pretrained_resampler_name_or_path
        resampler_params = vip_params.resampler_params

    load_dtype = torch.bfloat16 


    device = torch.device(f"cuda:{gpu_id}")

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.get("revision", None),
        variant=args.get("variant", None),
    ).to(device)

    if use_vip:
        transformer.set_vip_layers(vip_path, **vip_params)
        transformer = transformer.to(dtype)

        resampler = Resampler.from_pretrained(
            vip_path,
            subfolder="resampler",
            torch_dtype=dtype
        ).to(device)
        resampler.set_pca(args.get("longvgen_pca", None))

    else:
        resampler = None

    pipe = MPFIFOVideoIPAdapterCogVideoXPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer,
        resampler=resampler,
        torch_dtype=dtype
    )
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    output_queue.put((gpu_id, pipe))
    del transformer
    del resampler

def init_pipeline(gpu_id, args, dtype):
    use_vip = args.use_vip
    if use_vip:
        vip_params = args.video_ipadapter_params
        use_vae_as_encoder = vip_params.use_vae_as_encoder
        vip_path = args.pretrained_resampler_name_or_path
        resampler_params = vip_params.resampler_params

    load_dtype = torch.bfloat16 

    device = torch.device(f"cuda:{gpu_id}")

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.get("revision", None),
        variant=args.get("variant", None),
    ).to(device)

    if use_vip:
        transformer.set_vip_layers(vip_path, **vip_params)
        transformer = transformer.to(dtype)

        resampler = Resampler.from_pretrained(
            vip_path,
            subfolder="resampler",
            torch_dtype=dtype
        ).to(device)
        resampler.set_pca(args.get("longvgen_pca", None))

    else:
        resampler = None

    pipe = MPFIFOVideoIPAdapterCogVideoXPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer,
        resampler=resampler,
        torch_dtype=dtype
    )
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    return pipe


def main(args):
    *_, config = inspect.getargvalues(inspect.currentframe())
    # create folder
    args.output_dir = create_output_folders(args.output_dir, config, args.name_prefix)

    args.gpu = list(range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))))
    print(f"Running on gpus: {args.gpu}")

    dtype=torch.float32
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16

    ###################################################
    ######## Prepare 1st models and scheduler #########
    ###################################################

    use_vip = args.use_vip
    if use_vip:
        vip_params = args.video_ipadapter_params
        use_vae_as_encoder = vip_params.use_vae_as_encoder
        vip_path = args.pretrained_resampler_name_or_path
        resampler_params = vip_params.resampler_params

    pipe_list = []
    for gpu_id in args.gpu:
        pipe_list.append(init_pipeline(gpu_id, args, dtype))
        
    ###################################################
    ######## Prepare 2nd models and scheduler #########
    ###################################################

    if args.use_2nd_stage:
        tokens_transformer = CogVideoXTransformer3DModel.from_pretrained(
            args.pretrained_2nd_stage_model_name_or_path,
            subfolder="transformer",
            torch_dtype=dtype
        )
        pipe_2nd = LongVGenCogVideoXPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=tokens_transformer,
            torch_dtype=dtype
        )
        pipe_2nd.scheduler = CogVideoXDPMScheduler.from_config(
            pipe_2nd.scheduler.config,
            timestep_spacing="trailing"
        )
        pipe_2nd.to(f"cuda:{args.gpu[0]}")
        
    inputs = args.input_config
    public_dps = inputs.pop("public")
    items_json = inputs.pop("input_json") if inputs.get("input_json") is not None else None
    if items_json is not None:
        with open(items_json, "r") as f:
            items = json.load(f)
        inputs.update(items.get("input_config"))

    # inference!
    print("***** Running inference *****")
    print(f"  Num items = {len(inputs)}")

    progress_bar = tqdm(
        range(0, len(inputs)),
        initial=0,
        desc="Steps",
    )
        
    for name, item in inputs.items():
        prompt = item["prompt"]
        print(f"Processing {name}: [{prompt}]")

        dps = copy.deepcopy(public_dps)
        dps.update(item.get("params", {}))

        video = None
        image_embeddings = None

        if args.use_vip:
            if args.use_2nd_stage:
                # calculate image embeddings
                image_embeddings = pipe_2nd(
                    prompt=prompt,
                    height=resampler_params.num_height_queries,
                    width=resampler_params.num_width_queries,
                    num_frames_per_chunk=resampler_params.num_temporal_queries,
                    num_chunks=dps.max_num_chunks,
                    use_dynamic_cfg=True,
                    guidance_scale=args.get("guidance_scale_2nd", args.guidance_scale),
                    generator=torch.Generator().manual_seed(args.seed_2nd),
                    longvgen_mean=args.longvgen_mean,
                    longvgen_std=args.longvgen_std,
                    longvgen_pca=args.longvgen_pca,
                ).frames
                #print("image_embeddings", image_embeddings.shape)
                #print("image_embeddings", image_embeddings.min(), image_embeddings.max(), image_embeddings.mean())

            else:
                video_path = item.get("video")
                assert video_path is not None

                video = load_video(
                    video_path,
                    dps.output_res,
                    args.num_frames_per_chunk,
                    dps.pad_to_fit,
                    dps.sample_fps,
                    dps.start_t,
                    dps.end_t,
                    dps.max_num_chunks,
                    dps.crop_to_fit
                )
                #print(video.shape, video.device)

        base_outputs = pipe_list[0](
            prompt=prompt,
            frames=video,
            image_embeddings=image_embeddings,
            num_videos_per_prompt=dps.num_videos_per_prompt,
            num_inference_steps=args.num_inference_steps,
            num_frames_per_chunk=args.num_frames_per_chunk,
            max_num_chunks=dps.max_num_chunks,
            max_num_chunks_wo_fifo=dps.max_num_chunks_wo_fifo,
            max_num_chunks_w_fifo=dps.max_num_chunks_w_fifo,
            use_dynamic_cfg=False,
            use_separate_guidance=dps.get("use_separate_guidance", args.get("use_separate_guidance", False)),
            guidance_scale=args.guidance_scale,
            guidance_scale_img=args.get("guidance_scale_img", args.guidance_scale),
            generator=torch.Generator().manual_seed(args.seed),
            vip_scale=vip_params.scale if args.use_vip else 1.0,
            sampling_mode=args.get("sampling_mode"),
            sampling_params=args.get("sampling_params"),
            cache_idx=args.get("cache_idx"),
            video_ipadapter_start_frame_idx=vip_params.video_ipadapter_start_frame_idx if args.use_vip else 1000,
            return_dict=False,
        )
        for i in range(1,len(pipe_list)):
            pipe_list[i].preprare_for_fifo(
            prompt=prompt,
            frames=video,
            image_embeddings=image_embeddings,
            num_videos_per_prompt=dps.num_videos_per_prompt,
            num_inference_steps=args.num_inference_steps,
            num_frames_per_chunk=args.num_frames_per_chunk,
            max_num_chunks=dps.max_num_chunks,
            max_num_chunks_wo_fifo=dps.max_num_chunks_wo_fifo,
            max_num_chunks_w_fifo=dps.max_num_chunks_w_fifo,
            use_dynamic_cfg=False,
            use_separate_guidance=dps.get("use_separate_guidance", args.get("use_separate_guidance", False)),
            guidance_scale=args.guidance_scale,
            guidance_scale_img=args.get("guidance_scale_img", args.guidance_scale),
            generator=torch.Generator().manual_seed(args.seed),
            vip_scale=vip_params.scale if args.use_vip else 1.0,
            sampling_mode=args.get("sampling_mode"),
            sampling_params=args.get("sampling_params"),
            cache_idx=args.get("cache_idx"),
            video_ipadapter_start_frame_idx=vip_params.video_ipadapter_start_frame_idx if args.use_vip else 1000,
            return_dict=False,
        )

        orig_video_frames, video_frames, cache_video_frames = cogvideo_fifo_mp_v2(
            pipe_list,
            base_outputs,
        )
                    
        if video is not None:
            export_to_video(
                (video.clamp(-1,1)/2+0.5).squeeze().permute(0,2,3,1).cpu().numpy(),
                os.path.join(args.output_dir, f"{name}_source_{prompt[:20]}.mp4"),
                fps=dps.output_fps
            )
        if image_embeddings is not None:
            torch.save(
                image_embeddings[0].cpu(),
                os.path.join(args.output_dir, f"{name}_embeds_{prompt[:20]}.pt")
            )
        export_to_video(
            orig_video_frames[0], 
            os.path.join(args.output_dir, f"{name}_orig_{prompt[:20]}.mp4"), 
            fps=dps.output_fps
        )
        export_to_video(
            video_frames[0],
            os.path.join(args.output_dir, f"{name}_fifo_{prompt[:20]}.mp4"),
            fps=dps.output_fps
        )
        if cache_video_frames is not None:
            for _cid, cache_video in enumerate(cache_video_frames):
                os.makedirs(os.path.join(args.output_dir, "cache"), exist_ok=True)
                out_file = os.path.join(
                    args.output_dir,
                    "cache",
                    f"{name}_cache_{args.cache_idx[_cid]}_{prompt[:20]}.mp4",
                )
                export_to_video(cache_video[0], out_file, dps.output_fps)
        progress_bar.update(1)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/infer/edit.yaml')
    args = parser.parse_args()
    main(OmegaConf.load(args.config))
