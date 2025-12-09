import PIL
from PIL import Image
from einops import rearrange, repeat

import torchvision.transforms as TT
from torchvision.transforms import Resize, Pad, InterpolationMode, ToTensor, InterpolationMode
from torchvision.transforms.functional import center_crop, resize

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ResolutionControl(object):

    def __init__(self, input_res, output_res, pad_to_fit=False, fill=0, **kwargs):
    
        self.ih, self.iw = input_res
        self.output_res = output_res
        self.pad_to_fit = pad_to_fit
        self.fill=fill
        
    def pad_with_ratio(self, frames, res, fill=0):
        if isinstance(frames, torch.Tensor):
            original_dim = frames.ndim
            if frames.ndim > 4:
                batch_size = frames.shape[0]
                frames = rearrange(frames, "b f c h w -> (b f) c h w")
            _, _, ih, iw = frames.shape
        elif isinstance(frames, PIL.Image.Image):
            iw, ih = frames.size
        assert ih == self.ih and iw == self.iw, "resolution doesn't match."
        #print("ih, iw", ih, iw)
        i_ratio = ih / iw
        h, w = res
        #print("h,w", h ,w)
        n_ratio = h / w
        if i_ratio > n_ratio:
            nw = int(ih / h * w)
            #print("nw", nw)
            frames = Pad(((nw - iw)//2,0), fill=fill)(frames)
        else:
            nh = int(iw / w * h)
            frames = Pad((0,(nh - ih)//2), fill=fill)(frames)
        #print("after pad", frames.shape)
        if isinstance(frames, torch.Tensor):
            if original_dim > 4:
                frames = rearrange(frames, "(b f) c h w -> b f c h w", b=batch_size)
        
        return frames

    def return_to_original_res(self, frames):
        if isinstance(frames, torch.Tensor):
            original_dim = frames.ndim
            if frames.ndim > 4:
                batch_size = frames.shape[0]
                frames = rearrange(frames, "b f c h w -> (b f) c h w")
            _, _, h, w = frames.shape
        elif isinstance(frames, PIL.Image.Image):
            w, h = frames.size
        #print("original res", (self.ih, self.iw))
        #print("current res", (h, w))
        assert h == self.output_res[0] and w == self.output_res[1], "resolution doesn't match."
        n_ratio = h / w
        ih, iw = self.ih, self.iw
        i_ratio = ih / iw
        if self.pad_to_fit:
            if i_ratio > n_ratio:
                nw = int(ih / h * w)
                frames = Resize((ih, iw+2*(nw - iw)//2), interpolation=InterpolationMode.BILINEAR, antialias=True)(frames)
                if isinstance(frames, torch.Tensor):
                    frames = frames[...,:,(nw - iw)//2:-(nw - iw)//2]
                elif isinstance(frames, PIL.Image.Image):
                    frames = frames.crop(((nw - iw)//2,0,iw+(nw - iw)//2,ih))              
            else:
                nh = int(iw / w * h)
                frames = Resize((ih+2*(nh - ih)//2, iw), interpolation=InterpolationMode.BILINEAR, antialias=True)(frames)
                if isinstance(frames, torch.Tensor):
                    frames = frames[...,(nh - ih)//2:-(nh - ih)//2,:]
                elif isinstance(frames, PIL.Image.Image):
                    frames = frames.crop((0,(nh - ih)//2,iw,ih+(nh - ih)//2))
        else:
            frames = Resize((ih, iw), interpolation=InterpolationMode.BILINEAR, antialias=True)(frames)

        if isinstance(frames, torch.Tensor):
            if original_dim > 4:
                frames = rearrange(frames, "(b f) c h w -> b f c h w", b=batch_size)

        return frames

    def __call__(self, frames):
        if self.pad_to_fit:
            frames = self.pad_with_ratio(frames, self.output_res, fill=self.fill)
        
        if isinstance(frames, torch.Tensor):
            original_dim = frames.ndim
            if frames.ndim > 4:
                batch_size = frames.shape[0]
                frames = rearrange(frames, "b f c h w -> (b f) c h w")
            frames = (frames + 1) / 2.

        frames = Resize(tuple(self.output_res), interpolation=InterpolationMode.BILINEAR, antialias=True)(frames)
        if isinstance(frames, torch.Tensor):
            if original_dim > 4:
                frames = rearrange(frames, "(b f) c h w -> b f c h w", b=batch_size)
            frames = frames * 2 - 1

        return frames

    def callback(self, frames):
        return self.return_to_original_res(frames)

def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr
