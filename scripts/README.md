# Uncovering the Disentanglement Capability in Text-to-Image Diffusion Models

[Qiucheng Wu](https://wuqiuche.github.io/)<sup>1</sup>,
[Yujian Liu](https://yujianll.github.io)<sup>1</sup>,
[Handong Zhao](https://hdzhao.github.io)<sup>2</sup>,
[Ajinkya Kale](https://dblp.org/pid/04/6453.html)<sup>2</sup>,
[Trung Bui](https://sites.google.com/site/trungbuistanford/)<sup>2</sup>,
[Tong Yu](https://dblp.org/pid/32/1593-1.html)<sup>2</sup>,
[Zhe Lin](https://dblp.uni-trier.de/pid/42/1680-1.html)<sup>2</sup>,
[Yang Zhang](https://mitibmwatsonailab.mit.edu/people/yang-zhang/)<sup>3</sup>,
[Shiyu Chang](https://code-terminator.github.io/)<sup>1</sup>
<br>
<sup>1</sup>University of California, Santa Barbara, <sup>2</sup>Adobe Research <sup>3</sup>MIT-IBM Watson AI Lab, IBM Research

This is the official implementation of the paper "Uncovering the Disentanglement Capability in Text-to-Image Diffusion Models".

## Overview
Generative models have been widely studied in computer vision. Recently, diffusion models have drawn substantial attention due to the high quality of their generated images. A key desired property of image generative models is the ability to disentangle different attributes, which should enable modification towards a style without changing the semantic content, and the modification parameters should generalize to different images. Previous studies have found that generative adversarial networks (GANs) are inherently endowed with such disentanglement capability, so they can perform disentangled image editing without re-training or fine-tuning the network. In this work, we explore whether diffusion models are also inherently equipped with such a capability. Our finding is that for stable diffusion models, by partially changing the input text embedding from a neutral description (e.g., "a photo of person") to one with style (e.g., "a photo of person with smile") while fixing all the Gaussian random noises introduced during the denoising process, the generated images can be modified towards the target style without changing the semantic content. Based on this finding, we further propose a simple, light-weight image editing algorithm where the mixing weights of the two text embeddings are optimized for style matching and content preservation. This entire process only involves optimizing over around 50 parameters and does not fine-tune the diffusion model itself. Experiments show that the proposed method can modify a wide range of attributes, with the performance outperforming diffusion-model-based image-editing algorithms that require fine-tuning. The optimized weights generalize well to different images.

![](./assets/teaser.png)

## The workflow
Here, we demonstrate an example of disentangling target attribute "children drawing". In this example, c(0) is the embedding of “A castle”, and c(1) is the embedding of “A children drawing of castle”. The first step  (*first two rows*) is the optimization process that finds the best soft combination of c(0) and c(1), such that the modified image (the second row) changes the attribute without affecting other contents. After this, the learned text embedding can be directly applied to a new image, which leads to the same editing effect (*last row*).

![](./assets/pipeline.png)

## Requirements
Our code is based on <a href="https://github.com/CompVis/stable-diffusion">stable-diffusion</a>. Please first download environment from their repo. You will also need to download the pretrained stable-diffusion model. The following bash script prepares the repository for you. This project requires one GPU with memory 48GB.
```bash
bash scripts/prepare.bash
```

## Disentangle Attributes
```bash
python scripts/disentangle.py --c1 <neutral_prompt> --c2 <target_prompt> --seed 42 --outdir <output_dir>
```
We provide a bash file with a disentangling example:
```bash
bash scripts/disentangle.bash
```

## Edit Images
```bash
python scripts/edit.py --c1 <neutral_prompt> --c2 <target_prompt> --seed 42 --input <input_image> --outdir <output_dir>
```
We provide a bash file with an image editing example:
```bash
bash scripts/edit.bash
```

## Replication
To replicate our results in paper, we provide a bash file with commands used. You can run them all at once, or choose the target attributes you are interested in.
```bash
bash result.bash
```

## Results
Our method is able to disentangle a series of global and local attributes. We demonstrate examples below. The high-resolution images can be found in ```examples``` directory.

![](./assets/example1.png)

![](./assets/example2.png)

## Parent Repository
This code is adopted from <a href="">https://github.com/CompVis/stable-diffusion</a> and <a href="">https://github.com/orpatashnik/StyleCLIP</a>.

