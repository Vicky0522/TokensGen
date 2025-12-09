import os
import decord
import imageio
import random
import numpy as np
import PIL
from PIL import Image
from einops import rearrange, repeat
from typing import Optional, List
import cv2
import time
random.seed(int(time.time()))

from torchvision.transforms import Resize, Pad, InterpolationMode, ToTensor, InterpolationMode

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import CLIPTokenizer
from huggingface_hub import hf_hub_download
from huggingface_hub import HfFileSystem

from .utils import ResolutionControl, resize_for_rectangle_crop

import pandas as pd

def load_video(
    video_path, output_res, nf_per_chunk, pad_to_fit,
    sample_fps, start_t, end_t, max_num_chunks,
    crop_to_fit=False
):
    # read video
    vr = decord.VideoReader(video_path)
    initial_fps = vr.get_avg_fps()

    if sample_fps == -1: sample_fps = initial_fps
    if end_t == -1:
        end_t = len(vr) / initial_fps
    else:
        end_t = min(len(vr) / initial_fps, end_t)
    assert 0 <= start_t < end_t
    assert sample_fps > 0

    start_f_ind = int(start_t * initial_fps)
    end_f_ind = int(end_t * initial_fps)
    num_f = int((end_t - start_t) * sample_fps) 
    sample_idx = np.linspace(start_f_ind, end_f_ind, num_f, endpoint=False).astype(int)
    num_chunks = min(len(sample_idx)//nf_per_chunk, max_num_chunks)
    sample_idx = sample_idx[:num_chunks * nf_per_chunk]
    assert len(sample_idx) > 0, f"sample_idx is empty!"

    video = vr.get_batch(sample_idx)
    if not isinstance(video, torch.Tensor):
        video = torch.Tensor(video.asnumpy()).type(torch.float)#/255.
    else:
        video = video.type(torch.float)#/255.

    print("read video", video.min(), video.max())

    video = rearrange(video, "f h w c -> f c h w")

    #if crop_to_fit:
    #    pixel_values = resize_for_rectangle_crop(video, output_res, reshape_mode="center")
    #    pixel_values = pixel_values / 127.5 - 1.0
    if crop_to_fit:
        pixel_values = video / 255.
        pixel_values = resize_for_rectangle_crop(pixel_values, output_res, reshape_mode="center")
        print("pixel_values", pixel_values.min(), pixel_values.max())
        pixel_values = pixel_values * 2 - 1
    else:
        video = video / 127.5 - 1.0
        resctrl = ResolutionControl(video.shape[-2:],output_res,pad_to_fit,fill=-1)
        pixel_values = resctrl(video)

    return pixel_values.unsqueeze(0)

    

class LongVideoDataset(Dataset):

    def __init__(
        self,
        video_dir: str,
        height: int,
        width: int,
        pad_to_fit: bool = False,
        start_t: int = 0,
        end_t: int = -1,
        sample_fps: int = -1,
        chunk_size: int = 14,
        max_num_chunks: int = 2,
        device: str = "cuda",
        tokenizer: Optional[CLIPTokenizer] = None,
        **kwargs 
    ):
        video_file_list = []
        for root, _, fnames in os.walk(video_dir):
            for fname in fnames:
                if fname.endswith(".mp4"):
                    video_file_list.append(os.path.join(root, fname))

        self.video_file_list = video_file_list
        self.video_dir = video_dir
        self.height = int(height)
        self.width = int(width)
        self.start_t = int(start_t)
        self.end_t = int(end_t)
        self.sample_fps = float(sample_fps)
        self.chunk_size = int(chunk_size)
        self.max_num_chunks = int(max_num_chunks)
        self.device = device
        self.pad_to_fit = pad_to_fit
        self.tokenizer = tokenizer


        
    @staticmethod
    def __getname__(): return 'long_video'

    def __len__(self):
        return len(self.video_file_list)

    def get_prompt_ids(self, prompt):
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    def __getitem__(self, index):
        video_path = self.video_file_list[index] 

        # read video
        vr = decord.VideoReader(video_path)
        initial_fps = vr.get_avg_fps()
        sample_fps = self.sample_fps
        start_t = self.start_t
        end_t = self.end_t

        if sample_fps == -1: sample_fps = initial_fps
        if end_t == -1:
            end_t = len(vr) / initial_fps
        else:
            end_t = min(len(vr) / initial_fps, end_t)
        assert 0 <= start_t < end_t
        assert sample_fps > 0

        start_f_ind = int(start_t * initial_fps)
        end_f_ind = int(end_t * initial_fps)
        num_f = int((end_t - start_t) * sample_fps) 
        sample_idx = np.linspace(start_f_ind, end_f_ind, num_f, endpoint=False).astype(int)
        num_chunks = min(len(sample_idx)//self.chunk_size, self.max_num_chunks)
        sample_idx = sample_idx[:num_chunks * self.chunk_size]
        assert len(sample_idx) > 0, f"sample_idx is empty!"

        video = vr.get_batch(sample_idx)
        if not isinstance(video, torch.Tensor):
            video = torch.Tensor(video.asnumpy()).type(torch.float)#/255.
        else:
            video = video.type(torch.float)#/255.

#        print("read video", video.min(), video.max())
        #video = video.to(self.device).to(self.dtype)
        # save for debug
#        print(video.shape, video.min(), video.max())
#        for i in range(0, 3):
#            frame = (video[i*self.chunk_size,...]*255.).type(torch.uint8).cpu().numpy()
#            print(frame.shape, frame.min(), frame.max())
#            cv2.imwrite(f"{i}.png", frame[:,:,::-1])
#        print("save done.")

        video = rearrange(video, "f h w c -> f c h w")
        video = video / 127.5 - 1.0

        resctrl = ResolutionControl(video.shape[-2:],(self.height,self.width),self.pad_to_fit,fill=-1)
        pixel_values = resctrl(video)

        motion_values = torch.Tensor([127.])

        outputs = {
            "pixel_values": pixel_values,
            "motion_values": motion_values,
            'dataset': self.__getname__(),
        }

        if self.tokenizer is not None:
            prompt = "A silver Jeep driving down a curvy road in the countryside."
            prompt_ids = self.get_prompt_ids(prompt)
            outputs.update({
                "prompt_ids": prompt_ids[0]
            })

        return outputs

class LongVideoSummaryDataset(Dataset):

    def __init__(
        self,
        video_dir: str,
        height: int,
        width: int,
        pad_to_fit: bool = False,
        start_t: int = 0,
        end_t: int = -1,
        sample_fps: int = -1,
        chunk_size: int = 14,
        max_num_chunks: int = 2,
        device: str = "cuda",
        tokenizer: Optional[CLIPTokenizer] = None,
        sort: bool = True,
        num: int = None,
        **kwargs 
    ):
        video_file_list = []
        for root, _, fnames in os.walk(video_dir):
            for fname in fnames:
                if fname.endswith(".mp4"):
                    video_file_list.append(os.path.join(root, fname))

        def filenum(x):
            x = os.path.basename(x).split(".")[0]
            n1 = int(x.split("_")[0])
            n2 = int(x.split("-")[1])
            return n1 * 1e6 + n2 

        if sort:
            video_file_list = sorted(video_file_list, key = filenum) 

        num = len(video_file_list) if num is None else num
        video_file_list = video_file_list[:num]

        self.video_file_list = video_file_list
        self.video_dir = video_dir
        self.height = int(height)
        self.width = int(width)
        self.start_t = int(start_t)
        self.end_t = int(end_t)
        self.sample_fps = float(sample_fps)
        self.chunk_size = int(chunk_size)
        self.max_num_chunks = int(max_num_chunks)
        self.device = device
        self.pad_to_fit = pad_to_fit
        self.tokenizer = tokenizer
        self.cache_file_list = None

    @staticmethod
    def __getname__(): return 'long_video'

    def __len__(self):
        return len(self.video_file_list)

    def load_cached_batch(self, video_cache_list_path):
        video_cache_list = []
        with open(video_cache_list_path, "r") as f:
            for line in f:
                video_cache_list.append(line.strip().split(" "))

        num = len(self.video_file_list)
        self.video_file_list = [item[0] for item in video_cache_list][:num]
        self.cache_file_list = [item[1] for item in video_cache_list][:num]

    def get_prompt_ids(self, prompt):
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    def __getitem__(self, index):
        video_path = self.video_file_list[index] 

        # read video
        vr = decord.VideoReader(video_path)
        initial_fps = vr.get_avg_fps()
        sample_fps = self.sample_fps
        start_t = self.start_t
        end_t = self.end_t

        if sample_fps == -1: sample_fps = initial_fps
        if end_t == -1:
            end_t = len(vr) / initial_fps
        else:
            end_t = min(len(vr) / initial_fps, end_t)
        assert 0 <= start_t < end_t
        assert sample_fps > 0

        start_f_ind = int(start_t * initial_fps)
        end_f_ind = int(end_t * initial_fps)
        num_f = int((end_t - start_t) * sample_fps) 
        sample_idx = np.linspace(start_f_ind, end_f_ind, num_f, endpoint=False).astype(int)
        num_chunks = min((len(sample_idx)-1)//(self.chunk_size-1), self.max_num_chunks)
        sample_idx = sample_idx[:1+num_chunks * (self.chunk_size-1)]
        assert len(sample_idx) > 0, f"sample_idx is empty!"

        keyframe_idx = [0,]
        for i in range(num_chunks):
            keyframe_idx += [(i+1)*(self.chunk_size-1),]

        video = vr.get_batch(sample_idx)
        if not isinstance(video, torch.Tensor):
            video = torch.Tensor(video.asnumpy()).type(torch.float)/255.
        #else:
            #video = video.type(torch.float)/255.
        #print("dataset", video.min(), video.max())
        #video = video.to(self.device).to(self.dtype)
        # save for debug
#        print(video.shape, video.min(), video.max())
#        for i in range(0, 3):
#            frame = (video[i*self.chunk_size,...]*255.).type(torch.uint8).cpu().numpy()
#            print(frame.shape, frame.min(), frame.max())
#            cv2.imwrite(f"{i}.png", frame[:,:,::-1])
#        print("save done.")

        video = rearrange(video, "f h w c -> f c h w")
        video = video / 127.5 - 1.0

        resctrl = ResolutionControl(video.shape[-2:],(self.height,self.width),self.pad_to_fit,fill=-1)
        pixel_values = resctrl(video)

        motion_values = torch.Tensor([127.])

        outputs = {
            "pixel_values": pixel_values[keyframe_idx, ...],
            "refer_pixel_values": pixel_values[0:1,...],
            "cross_pixel_values": pixel_values,
            "motion_values": motion_values,
            'dataset': self.__getname__(),
            'video_path': video_path 
        }

        if self.tokenizer is not None:
            prompt = "It is a futuristic city with neon lights and flying cars."
            prompt_ids = self.get_prompt_ids(prompt)
            outputs.update({
                "prompt_ids": prompt_ids[0]
            })

        if self.cache_file_list is not None:
            cached_latent = torch.load(self.cache_file_list[index], map_location='cuda:0')
            for key, value in cached_latent.items():
                assert not torch.any(torch.isnan(value))
            outputs.update(cached_latent)

        return outputs

class LongVideoDatasetForVC2(Dataset):

    def __init__(
        self,
        video_dir: str,
        height: int,
        width: int,
        pad_to_fit: bool = False,
        start_t: int = 0,
        end_t: int = -1,
        sample_fps: int = -1,
        chunk_size: int = 14,
        max_num_chunks: int = 2,
        device: str = "cuda",
        tokenizer: Optional[CLIPTokenizer] = None,
        **kwargs 
    ):
        video_file_list = []
        for root, _, fnames in os.walk(video_dir):
            for fname in fnames:
                if fname.endswith(".mp4"):
                    video_file_list.append(os.path.join(root, fname))

        self.video_file_list = video_file_list
        self.video_dir = video_dir
        self.height = int(height)
        self.width = int(width)
        self.start_t = int(start_t)
        self.end_t = int(end_t)
        self.sample_fps = float(sample_fps)
        self.chunk_size = int(chunk_size)
        self.max_num_chunks = int(max_num_chunks)
        self.device = device
        self.pad_to_fit = pad_to_fit
        self.tokenizer = tokenizer

        
    @staticmethod
    def __getname__(): return 'long_video'

    def __len__(self):
        return len(self.video_file_list)

    def get_prompt_ids(self, prompt):
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    def __getitem__(self, index):
        video_path = self.video_file_list[index] 

        # read video
        vr = decord.VideoReader(video_path)
        initial_fps = vr.get_avg_fps()
        sample_fps = self.sample_fps
        start_t = self.start_t
        end_t = self.end_t

        if sample_fps == -1: sample_fps = initial_fps
        if end_t == -1:
            end_t = len(vr) / initial_fps
        else:
            end_t = min(len(vr) / initial_fps, end_t)
        assert 0 <= start_t < end_t
        assert sample_fps > 0

        start_f_ind = int(start_t * initial_fps)
        end_f_ind = int(end_t * initial_fps)
        num_f = int((end_t - start_t) * sample_fps) 
        sample_idx = np.linspace(start_f_ind, end_f_ind, num_f, endpoint=False).astype(int)
        num_chunks = min(len(sample_idx)//self.chunk_size, self.max_num_chunks)
        sample_idx = sample_idx[:num_chunks * self.chunk_size]
        assert len(sample_idx) > 0, f"sample_idx is empty!"

        video = vr.get_batch(sample_idx)
        if not isinstance(video, torch.Tensor):
            video = torch.Tensor(video.asnumpy()).type(torch.float)#/255.
        else:
            video = video.type(torch.float)#/255.

        #print("read video", video.min(), video.max())
        #video = video.to(self.device).to(self.dtype)
        # save for debug
#        print(video.shape, video.min(), video.max())
#        for i in range(0, 3):
#            frame = (video[i*self.chunk_size,...]*255.).type(torch.uint8).cpu().numpy()
#            print(frame.shape, frame.min(), frame.max())
#            cv2.imwrite(f"{i}.png", frame[:,:,::-1])
#        print("save done.")

        video = rearrange(video, "f h w c -> f c h w")
        video = video / 127.5 - 1.0

        resctrl = ResolutionControl(video.shape[-2:],(self.height,self.width),self.pad_to_fit,fill=-1)
        pixel_values = resctrl(video)
        pixel_values = rearrange(pixel_values, "f c h w -> c f h w")

        motion_values = torch.Tensor([127.])

        outputs = {
            "image": pixel_values,
            #"motion_values": motion_values,
            'dataset': self.__getname__(),
        }

        prompt = "A silver Jeep driving down a curvy road in the countryside."
        outputs.update({
            "caption": prompt
        })

        return outputs

class MiraDataset(Dataset):

    def __init__(
        self,
        csv_file: str,
        video_dir: str,
        height: int,
        width: int,
        pad_to_fit: bool = False,
        crop_to_fit: bool = False,
        start_t: int = 0,
        end_t: int = -1,
        random_sample: bool = False,
        random_flip: bool = False,
        sample_fps: int = -1,
        chunk_size: int = 14,
        max_num_chunks: int = 2,
        index: List[int] = None,
        device: str = "cuda",
        tokenizer: Optional[CLIPTokenizer] = None,
        i_drop_rate: float = 0.05,
        t_drop_rate: float = 0.05,
        ti_drop_rate: float = 0.05,
        use_frames_padding: bool = False,
        use_scene_detect: bool = False,
        scene_detect_file: str = None,
        **kwargs 
    ):
        

        self.video_dir = video_dir
        self.height = int(height)
        self.width = int(width)
        self.start_t = int(start_t)
        self.end_t = int(end_t)
        self.random_flip = random_flip
        self.random_sample = random_sample
        self.sample_fps = float(sample_fps)
        self.chunk_size = int(chunk_size)
        self.max_num_chunks = int(max_num_chunks)
        self.device = device
        self.pad_to_fit = pad_to_fit
        self.crop_to_fit = crop_to_fit
        self.tokenizer = tokenizer
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.use_frames_padding = use_frames_padding
        self.use_scene_detect = use_scene_detect


        video_csv = pd.read_csv(csv_file, encoding = "ISO-8859-1") 

        if index is None:
            index = [0, video_csv.shape[0]]
        index = list(index)

        if index[0] == -1:
            index[0] = 0
        if index[1] == -1:
            index[1] = video_csv.shape[0]

        self.video_csv = video_csv.loc[index[0]:index[1]-1]
        self.video_id = index

        if use_scene_detect:
            scene_detect = {}
            unqualified_video_list = []
            with open(scene_detect_file, "r") as f:
                for line in f:
                    results = line.split(" ")
                    if len(results) > 1 and len(results[1]) != 0:
                        video_name, scenes = results
                        _scene_detect = []
                        for _scene in scenes.split("|"):
                            start, end = _scene.split(",")
                            if int(end) - int(start) > self.max_num_chunks * self.chunk_size / self.sample_fps * 30:
                                _scene_detect.append([int(start), int(end)])
                        if len(_scene_detect) > 0:
                            scene_detect[video_name] = _scene_detect
                        else:
                            unqualified_video_list.append(video_name)

            self.scene_detect = scene_detect
            self.unqualified_video_list = unqualified_video_list
            print("unqualified_video_list:", unqualified_video_list)
            print("number of unqualified_video_list: ", len(unqualified_video_list))
        
    @staticmethod
    def __getname__(): return 'mira'

    def __len__(self):
        return self.video_csv.shape[0]

    def get_prompt_ids(self, prompt):
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    def __getitem__(self, index):
        flag = True
        while flag:
            index = index + self.video_id[0]
            video_index = self.video_csv.loc[index,"index"]
            if self.use_scene_detect:
                flag = (video_index in self.unqualified_video_list)
            else:
                flag = False
            if flag:
                index = random.randint(0, self.video_csv.shape[0]-1)

        dirn = int(video_index.split("_")[0]) // 1000
        video_path = os.path.join(self.video_dir, f"{dirn:09d}", video_index+".mp4") 

        # fetch scene detect results
        scenes = None
        if self.use_scene_detect:
            scenes = self.scene_detect.get(video_index, None)

        # read video
        vr = decord.VideoReader(video_path)
        initial_fps = vr.get_avg_fps()
        sample_fps = self.sample_fps
        start_t = self.start_t
        end_t = self.end_t

        if sample_fps == -1: sample_fps = initial_fps
        if end_t == -1:
            end_t = len(vr) / initial_fps
        else:
            end_t = min(len(vr) / initial_fps, end_t)
        assert 0 <= start_t < end_t
        assert sample_fps > 0


        if self.use_scene_detect and scenes is not None:
            assert len(scenes) > 0
            print("scenes", scenes)
            sample_idx_list = []
            random_idx_list = [0,]
            for scene in scenes:
                start_f_ind = scene[0]
                end_f_ind = scene[1]
                num_f = int((end_f_ind - start_f_ind) / initial_fps * sample_fps)
                sample_idx_list.append(
                    np.linspace(start_f_ind, end_f_ind, num_f, endpoint=False).astype(int)
                )
                assert len(sample_idx_list[-1]) >= self.chunk_size*self.max_num_chunks
                random_idx_list.append(num_f-self.chunk_size*self.max_num_chunks + 1 +random_idx_list[-1])
            print("before norm: random_idx_list", random_idx_list)
            random_idx_list = np.array([_item / max(1,random_idx_list[-1]) for _item in random_idx_list])
            sample_idx = sample_idx_list[0]
            print("random_idx_list", random_idx_list)
            # random sample for t
            if self.random_sample:
                rand_num = random.random()
                rand_list_ind = np.searchsorted(random_idx_list, rand_num, side='right') - 1 
                sample_idx = sample_idx_list[rand_list_ind]

        else:
            start_f_ind = int(start_t * initial_fps)
            end_f_ind = int(end_t * initial_fps)
            num_f = int((end_t - start_t) * sample_fps) 
            sample_idx = np.linspace(start_f_ind, end_f_ind, num_f, endpoint=False).astype(int)

        # random sample for t
        start_idx = 0
        if self.random_sample:
            start_idx = random.randint(0, len(sample_idx) - self.chunk_size * self.max_num_chunks)
            sample_idx = sample_idx[start_idx:]

        # calculate start_idx
        compressed_chunk_size = (self.chunk_size - 1) // 4 + 1
        start_idx = start_idx // self.chunk_size * compressed_chunk_size + \
                    int((start_idx%self.chunk_size) / float(self.chunk_size-1) * (compressed_chunk_size-1))

        num_chunks = min(len(sample_idx)//self.chunk_size, self.max_num_chunks)
        sample_idx = sample_idx[:num_chunks * self.chunk_size]
        assert len(sample_idx) > 0, f"sample_idx is empty!"

        video = vr.get_batch(sample_idx)
        if not isinstance(video, torch.Tensor):
            video = torch.Tensor(video.asnumpy()).type(torch.float)#/255.
        else:
            video = video.type(torch.float)#/255.

#        print("read video", video.min(), video.max())
        #video = video.to(self.device).to(self.dtype)
        # save for debug
#        print(video.shape, video.min(), video.max())
#        for i in range(0, 3):
#            frame = (video[i*self.chunk_size,...]*255.).type(torch.uint8).cpu().numpy()
#            print(frame.shape, frame.min(), frame.max())
#            cv2.imwrite(f"{i}.png", frame[:,:,::-1])
#        print("save done.")

        video = rearrange(video, "f h w c -> f c h w")
        if self.crop_to_fit:
            pixel_values = video / 255.
            pixel_values = resize_for_rectangle_crop(pixel_values, [self.height, self.width], reshape_mode="center")
            pixel_values = pixel_values * 2 - 1
        else:
            video = video / 127.5 - 1.0
            resctrl = ResolutionControl(video.shape[-2:],(self.height,self.width),self.pad_to_fit,fill=-1)
            pixel_values = resctrl(video)

        # random flip
        if self.random_flip:
            if random.random() < 0.5:
                pixel_values = pixel_values.flip((-1,))

        if self.use_frames_padding:
            pixel_values = torch.cat(
                [pixel_values,] + [pixel_values[[-1]],] * self.chunk_size * (self.max_num_chunks - num_chunks),
                dim=0
            )
            valid_num_chunks = num_chunks

        prompt = self.video_csv.loc[index,"dense_caption"]

        # classifier-free guidance
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            prompt = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            prompt = ""
            drop_image_embed = 1

        outputs = {
            "pixel_values": pixel_values,
            "prompt": prompt,
            "start_frame_idx": start_idx, 
            "video_index": video_index,
            "drop_image_embed": drop_image_embed,
            'dataset': self.__getname__(),
        }

        if self.tokenizer is not None:
            #prompt = "A silver Jeep driving down a curvy road in the countryside."
            prompt_ids = self.get_prompt_ids(prompt)
            outputs.update({
                "prompt_ids": prompt_ids[0],
            })

        if self.use_frames_padding:
            outputs.update({
                "valid_num_chunks": valid_num_chunks
            })

        return outputs

class LongVGenMiraDataset(Dataset):

    def __init__(
        self,
        csv_file: str,
        video_dir: str,
        index: List[int] = None,
        i_drop_rate: float = 0.05,
        t_drop_rate: float = 0.05,
        ti_drop_rate: float = 0.05,
        tokenizer: Optional[CLIPTokenizer] = None,
        **kwargs 
    ):

        self.video_dir = video_dir
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.tokenizer = tokenizer

        video_csv = pd.read_csv(csv_file, encoding = "ISO-8859-1") 

        if index is None:
            index = [0, video_csv.shape[0]]
        index = list(index)

        if index[0] == -1:
            index[0] = 0
        if index[1] == -1:
            index[1] = video_csv.shape[0]

        self.video_csv = video_csv.loc[index[0]:index[1]-1]
        self.video_id = index
        
    @staticmethod
    def __getname__(): return 'mira'

    def __len__(self):
        return self.video_csv.shape[0]

    def get_prompt_ids(self, prompt):
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    def __getitem__(self, index):
        index = index + self.video_id[0]
        video_index = self.video_csv.loc[index,"index"]
        dirn = int(video_index.split("_")[0]) // 1000
        video_path = os.path.join(self.video_dir, f"{dirn:09d}", video_index+".mp4") 

        prompt = self.video_csv.loc[index,"dense_caption"]

        # classifier-free guidance
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            prompt = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            prompt = ""
            drop_image_embed = 1

        outputs = {
            "prompt": prompt,
            "video_path": video_path,
            "video_index": video_index,
            "drop_image_embed": drop_image_embed,
            'dataset': self.__getname__(),
        }

        if self.tokenizer is not None:
            #prompt = "A silver Jeep driving down a curvy road in the countryside."
            prompt_ids = self.get_prompt_ids(prompt)
            outputs.update({
                "prompt_ids": prompt_ids[0],
            })

        return outputs

class VideoBatchDataset(Dataset):

    def __init__(
        self,
        video_path: List[str],
        height: int,
        width: int,
        pad_to_fit: bool = False,
        crop_to_fit: bool = False,
        start_t: int = 0,
        end_t: int = -1,
        random_sample: bool = False,
        random_flip: bool = False,
        sample_fps: int = -1,
        chunk_size: int = 14,
        max_num_chunks: int = 2,
        index: List[int] = None,
        device: str = "cuda",
        use_frames_padding: bool = False,
        **kwargs 
    ):

        self.video_paths = video_path
        self.height = int(height)
        self.width = int(width)
        self.start_t = int(start_t)
        self.end_t = int(end_t)
        self.random_flip = random_flip
        self.random_sample = random_sample
        self.sample_fps = float(sample_fps)
        self.chunk_size = int(chunk_size)
        self.max_num_chunks = int(max_num_chunks)
        self.device = device
        self.pad_to_fit = pad_to_fit
        self.crop_to_fit = crop_to_fit
        self.use_frames_padding = use_frames_padding

        def get_sample_idx(video_path): 
            # read video
            vr = decord.VideoReader(video_path)
            initial_fps = vr.get_avg_fps()
            sample_fps = self.sample_fps
            start_t = self.start_t
            end_t = self.end_t

            if sample_fps == -1: sample_fps = initial_fps
            if end_t == -1:
                end_t = len(vr) / initial_fps
            else:
                end_t = min(len(vr) / initial_fps, end_t)
            assert 0 <= start_t < end_t
            assert sample_fps > 0

            start_f_ind = int(start_t * initial_fps)
            end_f_ind = int(end_t * initial_fps)
            num_f = int((end_t - start_t) * sample_fps) 
            sample_idx = np.linspace(start_f_ind, end_f_ind, num_f, endpoint=False).astype(int)

            # random sample for t
            start_idx = 0
            if self.random_sample:
                start_idx = random.randint(0, max(0,len(sample_idx) - self.chunk_size * self.max_num_chunks - 1))
                sample_idx = sample_idx[start_idx:]

            # calculate start_idx
            compressed_chunk_size = (self.chunk_size - 1) // 4 + 1
            start_idx = start_idx // self.chunk_size * compressed_chunk_size + \
                        int((start_idx%self.chunk_size) / float(self.chunk_size-1) * (compressed_chunk_size-1))

            num_chunks = min(len(sample_idx)//self.chunk_size, self.max_num_chunks)
            sample_idx = sample_idx[:num_chunks * self.chunk_size]
            assert len(sample_idx) > 0, f"sample_idx is empty!"

            return sample_idx, start_idx

        self.sample_idxs = []
        self.start_idxs = []

        for _path in self.video_paths:
            _sample_idx, _start_idx = get_sample_idx(_path)
            self.sample_idxs.append(_sample_idx)
            self.start_idxs.append(_start_idx)
        
    @staticmethod
    def __getname__(): return 'videobatch'

    def __len__(self):
        return self.max_num_chunks

    def __getitem__(self, index):
        # read video
        def read_video(video_path, sample_idx):
            if index > len(sample_idx) // self.chunk_size - 1:
                return torch.zeros((self.chunk_size, 3, self.height, self.width)) 

            vr = decord.VideoReader(video_path)
            video = vr.get_batch(sample_idx[index*self.chunk_size:(index+1)*self.chunk_size])
            if not isinstance(video, torch.Tensor):
                video = torch.Tensor(video.asnumpy()).type(torch.float)#/255.
            else:
                video = video.type(torch.float)#/255.

            video = rearrange(video, "f h w c -> f c h w")
            if self.crop_to_fit:
                pixel_values = video / 255.
                pixel_values = resize_for_rectangle_crop(pixel_values, [self.height, self.width], reshape_mode="center")
                pixel_values = pixel_values * 2 - 1
            else:
                video = video / 127.5 - 1.0
                resctrl = ResolutionControl(video.shape[-2:],(self.height,self.width),self.pad_to_fit,fill=-1)
                pixel_values = resctrl(video)

            # random flip
            if self.random_flip:
                if random.random() < 0.5:
                    pixel_values = pixel_values.flip((-1,))

            return pixel_values

        pixel_values = []
        valid_num_chunks = []
        for video_path, sample_idx in zip(self.video_paths, self.sample_idxs):
            pixel_values.append(read_video(video_path, sample_idx))
            valid_num_chunks.append(len(sample_idx) // self.chunk_size)
        pixel_values = torch.stack(pixel_values)

        outputs = {
            "pixel_values": pixel_values,
            "valid_num_chunks": valid_num_chunks,
            "start_frame_idx": self.start_idxs,
            "index": index
        }

        return outputs

class VIPMiraDataset(Dataset):

    def __init__(
        self,
        csv_file: str,
        video_dir: str,
        height: int,
        width: int,
        pad_to_fit: bool = False,
        crop_to_fit: bool = False,
        start_t: int = 0,
        end_t: int = -1,
        random_sample: bool = False,
        random_flip: bool = False,
        sample_fps: int = -1,
        chunk_size: int = 14,
        max_num_chunks: int = 2,
        index: List[int] = None,
        device: str = "cuda",
        tokenizer: Optional[CLIPTokenizer] = None,
        i_drop_rate: float = 0.05,
        t_drop_rate: float = 0.05,
        ti_drop_rate: float = 0.05,
        use_frames_padding: bool = False,
        use_scene_detect: bool = False,
        scene_detect_file: str = None,
        **kwargs 
    ):

        self.video_dir = video_dir
        self.height = int(height)
        self.width = int(width)
        self.start_t = int(start_t)
        self.end_t = int(end_t)
        self.random_flip = random_flip
        self.random_sample = random_sample
        self.sample_fps = float(sample_fps)
        self.chunk_size = int(chunk_size)
        self.max_num_chunks = int(max_num_chunks)
        self.device = device
        self.pad_to_fit = pad_to_fit
        self.crop_to_fit = crop_to_fit
        self.tokenizer = tokenizer
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.use_frames_padding = use_frames_padding
        self.use_scene_detect = use_scene_detect


        video_csv = pd.read_csv(csv_file, encoding = "ISO-8859-1") 
        video_dict = {}
        for i in range(video_csv.shape[0]):
            video_index = video_csv.loc[i,"index"] 
            prompt = video_csv.loc[i, "dense_caption"]
            video_dict[video_index] = prompt

        fs = HfFileSystem()
        #vip_latents_list = fs.ls("datasets/Vicky0522/VIP/4x8x12_1300", detail=False) 
        vip_latents_list = []
        with open("good_samples_v2_train.txt", "r") as f:
            for line in f:
                vip_latents_list.append(line.strip())

        if index is None:
            index = [0, len(vip_latents_list)]
        index = list(index)

        if index[0] == -1:
            index[0] = 0
        if index[1] == -1:
            index[1] = len(vip_latents_list)

        self.video_dict = video_dict
#        self.video_csv = video_csv
        self.video_id = index
        self.vip_latents_list = vip_latents_list[index[0]:index[1]]
        self.fs = fs

    @staticmethod
    def __getname__(): return 'mira'

    def __len__(self):
        return len(self.vip_latents_list)

    def get_prompt_ids(self, prompt):
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    def __getitem__(self, index):
        video_pt = os.path.basename(self.vip_latents_list[index])
        video_index = video_pt[:-7]
        num_chunks = int(video_pt[-5:-3])
        prompt = self.video_dict[video_index]

#        video_vip_file = hf_hub_download(
#            repo_id="Vicky0522/VIP", 
#            filename=os.path.join("4x8x12_1300", video_pt), 
#            repo_type="dataset",
#            local_dir="./trash"
#        )
        flag = True
        while flag:
            try:
                with self.fs.open(self.vip_latents_list[index], "rb") as f:                
                    video = torch.load(f, weights_only=True)
#                video = torch.load(video_vip_file, weights_only=True)
#                os.system(f"rm -f {video_vip_file}")
                flag = False
            except Exception as e:
                print(e)
                time.sleep(1)
         
        video = video.type(torch.float)#/255.

        num_frames, channels, height, width = video.shape
        pixel_values = torch.zeros((self.max_num_chunks*num_frames//num_chunks, channels, height, width))
        pixel_values[:num_frames] = video

        # classifier-free guidance
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            prompt = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            prompt = ""
            drop_image_embed = 1

        outputs = {
            "pixel_values": pixel_values,
            "prompt": prompt,
            "start_frame_idx": 0, 
            "valid_num_chunks": num_chunks,
            "video_index": video_index,
            "drop_image_embed": drop_image_embed,
#            "download_path": video_vip_file,
            'dataset': self.__getname__(),
        }

        if self.tokenizer is not None:
            #prompt = "A silver Jeep driving down a curvy road in the countryside."
            prompt_ids = self.get_prompt_ids(prompt)
            outputs.update({
                "prompt_ids": prompt_ids[0],
            })

        return outputs

class VAEMiraDataset(Dataset):

    def __init__(
        self,
        csv_file: str,
        height: int,
        width: int,
        video_dir: str = "./datasets/MiraData/miradata/vae_latents/",
        pad_to_fit: bool = False,
        crop_to_fit: bool = False,
        start_t: int = 0,
        end_t: int = -1,
        random_sample: bool = False,
        random_flip: bool = False,
        sample_fps: int = -1,
        chunk_size: int = 14,
        max_num_chunks: int = 2,
        index: List[int] = None,
        device: str = "cuda",
        tokenizer: Optional[CLIPTokenizer] = None,
        i_drop_rate: float = 0.05,
        t_drop_rate: float = 0.05,
        ti_drop_rate: float = 0.05,
        use_frames_padding: bool = False,
        use_scene_detect: bool = False,
        scene_detect_file: str = None,
        **kwargs 
    ):

        self.video_dir = video_dir
        self.height = int(height)
        self.width = int(width)
        self.start_t = int(start_t)
        self.end_t = int(end_t)
        self.random_flip = random_flip
        self.random_sample = random_sample
        self.sample_fps = float(sample_fps)
        self.chunk_size = int(chunk_size)
        self.max_num_chunks = int(max_num_chunks)
        self.device = device
        self.pad_to_fit = pad_to_fit
        self.crop_to_fit = crop_to_fit
        self.tokenizer = tokenizer
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.use_frames_padding = use_frames_padding
        self.use_scene_detect = use_scene_detect


        video_csv = pd.read_csv(csv_file, encoding = "ISO-8859-1") 
        video_dict = {}
        for i in range(video_csv.shape[0]):
            video_index = video_csv.loc[i,"index"] 
            prompt = video_csv.loc[i, "dense_caption"]
            video_dict[video_index] = prompt

#        fs = HfFileSystem()
#        #vip_latents_list = fs.ls("datasets/Vicky0522/VIP/4x8x12_1300", detail=False) 
#        vip_latents_list = []
#        with open("good_samples_v2_train.txt", "r") as f:
#            for line in f:
#                vip_latents_list.append(line.strip())
        
        vae_latents_list = [os.path.join(video_dir, item) for item in os.listdir(video_dir)]

        if index is None:
            index = [0, len(vae_latents_list)]
        index = list(index)

        if index[0] == -1:
            index[0] = 0
        if index[1] == -1:
            index[1] = len(vae_latents_list)

        self.video_dict = video_dict
#        self.video_csv = video_csv
        self.video_id = index
        self.vae_latents_list = vae_latents_list[index[0]:index[1]]

    @staticmethod
    def __getname__(): return 'mira'

    def __len__(self):
        return len(self.vae_latents_list)

    def get_prompt_ids(self, prompt):
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    def __getitem__(self, index):
        video_pt = os.path.basename(self.vae_latents_list[index])
        video_index = video_pt[:-11]
        num_chunks = int(video_pt[-5:-3])
        prompt = self.video_dict[video_index]
        
        video = torch.load(self.vae_latents_list[index], weights_only=True)

        video = video.type(torch.float)#/255.

        num_frames, channels, height, width = video.shape
        pixel_values = torch.zeros((self.max_num_chunks*num_frames//num_chunks, channels, height, width))
        pixel_values[:num_frames] = video

        # classifier-free guidance
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            prompt = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            prompt = ""
            drop_image_embed = 1

        outputs = {
            "pixel_values": pixel_values,
            "prompt": prompt,
            "start_frame_idx": 0, 
            "valid_num_chunks": num_chunks,
            "video_index": video_index,
            "drop_image_embed": drop_image_embed,
#            "download_path": video_vip_file,
            'dataset': self.__getname__(),
        }

        if self.tokenizer is not None:
            #prompt = "A silver Jeep driving down a curvy road in the countryside."
            prompt_ids = self.get_prompt_ids(prompt)
            outputs.update({
                "prompt_ids": prompt_ids[0],
            })

        return outputs
