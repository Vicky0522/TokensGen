import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import os
import lpips
from torchvision.transforms import transforms

from longvgen.metrics.metric_util import reorder_image, to_y_channel

def calculate_lpips(img, img2, crop_border, input_order='CHW', test_y_channel=False, model = None, **kwargs):
#    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
#    if input_order not in ['HWC', 'CHW']:
#        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
#    img = reorder_image(img, input_order=input_order)
#    img2 = reorder_image(img2, input_order=input_order)
#
#    if crop_border != 0:
#        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
#        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
#
#    if test_y_channel:
#        img = to_y_channel(img)
#        img2 = to_y_channel(img2)
#
#    img = img.astype(np.float32)
#    img2 = img2.astype(np.float32)

    #loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
    loss_fn_vgg = model
    with torch.no_grad():
        # gt: convert rgb->lab->rgb
        #gt_lab = rgb_to_lab(gt)
        #gt_rgb = lab_to_rgb(gt_lab)

    #    img2 = transforms.ToTensor()(img2)
    #    img2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img2).cuda()
        #gt_rgb = transforms.ToTensor()(gt_rgb)
        #gt_rgb = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(gt_rgb).cuda()

    #    img = transforms.ToTensor()(img)
    #    img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img).cuda()

        lpips_vgg = loss_fn_vgg(img2, img).cpu()
        #lpips_vgg_convert = loss_fn_vgg(gt_rgb, pred).cpu()
    return lpips_vgg
