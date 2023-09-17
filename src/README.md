# Dataset generation

This directory contains the source code to generate SD-ImageNet, Complex SD-ImageNet, and SD-COCO.
The source code is based on [stable diffusion's txt2img.py](https://github.com/CompVis/stable-diffusion/blob/main/scripts/txt2img.py).

* Clone `https://github.com/moskomule/stable-diffusion`.
* Check if the commit id is `69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc`.
* Install dependencies accordingly. We used `torch==1.12.1@cuda11.3`.
* Copy the weight of stable diffusion to the `stable_diffusion` directory. We used `sd-v1-1.ckpt`.
* Run scripts.