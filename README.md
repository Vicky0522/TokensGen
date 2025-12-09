<br>
<p align="center">

<h1 align="center"><strong>TokensGen: Harnessing Condensed Tokens for Long Video Generation</strong></h1>
  <p align="center"><span><a href=""></a></span>
              <a href="https://vicky0522.github.io/Wenqi-Ouyang/">Wenqi Ouyang<sup>1</sup></a>
              <a href="https://xizaoqu.github.io">Zeqi Xiao<sup>1</sup></a>
              <a href="https://scholar.google.com/citations?user=qDsgBJAAAAAJ&hl=zh-CN">Danni Yang<sup>2</sup></a>
              <a href="https://zhouyifan.net/about/">Yifan Zhou<sup>1</sup></a>
              <a href="https://williamyang1991.github.io/">Shuai Yang<sup>3</sup></a>
              <a href="https://scholar.google.com/citations?user=jZH2IPYAAAAJ&hl=en">Lei Yang<sup>2</sup></a>
              <a href="https://jianlou.github.io/">Jianlou Si<sup>2</sup></a>
              <a href="https://xingangpan.github.io/">Xingang Pan<sup>1</sup></a>    <br>
    <sup>1</sup>S-Lab, Nanyang Technological University, <sup>2</sup>SenseTime Research <br> <sup>3</sup>Wangxuan Institute of Computer Technology, Peking University<br>  
    </p>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2507.15728" target='_blank'>
    <img src="https://img.shields.io/badge/arXiv-2507.15728-blue?">
  </a>
  <a href="https://vicky0522.github.io/tokensgen-webpage/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-&#x1F680-blue">
  </a>
</p>

> The official repo for "TokensGen: Harnessing Condensed Tokens for Long Video Generation".


https://github.com/user-attachments/assets/56279275-bcdf-460a-9e05-f9a589d7c020


## 🔥 News
* [2025-12-09] Our code and weights have been released
* [2025-07-20] Our project page has been established
* [2025-06-26] Our paper is accepted to ICCV 2025

## 🔧 TODO
- [x] Release code and weights

## 🧐 Methods
Overview of the model. Left: Overall Framework for TokensGen. Right: Trainable Modules.

<p align="center">
  <img src="assets/pipeline.jpg" width="100%">
</p>

## 🌿 Installation with conda

```
git clone https://github.com/Vicky0522/TokensGen.git
cd TokensGen
# install required packages
conda env create -f environment.yml
# install longvgen
python setup.py develop
```

## 🚀 Quick Start

### Download the weights
Download [CogVideoX-5b](https://huggingface.co/zai-org/CogVideoX-5b), [To2V](https://huggingface.co/Vicky0522/TokensGen-To2V) and [T2To](https://huggingface.co/Vicky0522/TokensGen-T2To). Place them under the created folder `weights/`. 

### Editing (To2V)
```
# single-GPU inference
CUDA_VISIBLE_DEVICES=0 python infer_cogvideo_mp_fifo.py --config config/infer/edit.yaml

# multi-GPU inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python infer_cogvideo_mp_fifo.py --config config/infer/edit.yaml
```

### Generation (T2To + To2V)
```
# single-GPU inference
CUDA_VISIBLE_DEVICES=0 python infer_cogvideo_mp_fifo.py --config config/infer/gen.yaml

# multi-GPU inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python infer_cogvideo_mp_fifo.py --config config/infer/gen.yaml
```

## ⚙️ Training

### Download the Dataset
Download the [MiraData](https://github.com/mira-space/MiraData)

### Train the To2V and T2To Model
Change the `video_dir` and `csv_file` in the YAML file to your own paths. We provide the [CSV](https://huggingface.co/datasets/Vicky0522/TokensGen-MiraData-CSV-Files) files that were used to train the T2To Model. 
After setting the paths, run the following command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_cogvideo_to2v.py --config config/train/cogvideo_5b_vaevip_4x8x12_to2v.yaml

# After the training of To2V finishes, run the data processing script to obtain VAE latents for long videos (only the selected videos are calculated):
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch calculate_vae_latents.py --config config/dataprocess/cogvideo_5b_vaevip_4x8x12_calculate_vae_latents.yaml

# Change the video_dir to the path of calculated VAE latents, and train the T2To Model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_cogvideo_t2to.py --config config/train/cogvideo_5b_vaevip_4x8x12_t2to.yaml
```

## ✏️ Citation

If our work is helpful for your research, please consider citing: 
```
@inproceedings{ouyang2025tokensgen,
  title={TokensGen: Harnessing Condensed Tokens for Long Video Generation},
  author={Ouyang, Wenqi and Xiao, Zeqi and Yang, Danni and Zhou, Yifan and Yang, Shuai and Yang, Lei and Si, Jianlou and Pan, Xingang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={18197--18206},
  year={2025}
}
```
