# [Will Large-scale Generative Models Corrupt Future Datasets?](https://arxiv.org/abs/2211.08095)

## Abstract

 Recently proposed large-scale text-to-image generative models such as DALL⋅E 2, Midjourney, and StableDiffusion can generate high-quality and realistic images from users' prompts. Not limited to the research community, ordinary Internet users enjoy these generative models, and consequently, a tremendous amount of generated images have been shared on the Internet. Meanwhile, today's success of deep learning in the computer vision field owes a lot to images collected from the Internet. These trends lead us to a research question: "**will such generated images impact the quality of future datasets and the performance of computer vision models positively or negatively?**" This paper empirically answers this question by simulating contamination. Namely, we generate ImageNet-scale and COCO-scale datasets using a state-of-the-art generative model and evaluate models trained with "contaminated" datasets on various tasks, including image classification and image generation. Throughout experiments, we conclude that generated images negatively affect downstream performance, while the significance depends on tasks and the amount of generated images. The generated datasets and the codes for experiments will be publicly released for future research.

## Datasets

The datasets (SD-ImageNet, Complex SD-ImageNet, and SD-COCO) can be downloaded from [this repository](https://dmsgrdm.riken.jp/c24kn/).
Before downloading them, check the LICENSE (CreativeML Open RAIL-M).

## Source Code

[src](./src) contains the source code to generate datasets.

## Citation

```bibtex
@inproceedings{hataya2023contamination,
  author = {Hataya, Ryuichiro and Bao, Han and Arai, Hiromi},
  title = {Will Large-scale Generative Models Corrupt Future Datasets?},
  booktitle = {ICCV},
  year = {2023},
}
```
