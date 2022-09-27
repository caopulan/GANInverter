# GAN Inverter

## News 
Code will be released soon (October).

GAN Inverter v1.0 integrates the following methods:
- [x] Encoder-based inversion: pSp, e4e
- [x] Optimization-based inversion: StyleGAN2-Projection
- [x] Hybrid methods: HFGI
- [x] **LSAP (under review at ICLR2023)**

## Introduction
A GAN inversion toolbox based on pytorch library.

We collect sota inversion methods and construct a uniform pipeline with more features. Inversion and editing are focused in this toolbox, and we have no plan to support other tasks which inversion method are not sota such as super-resolution, style-transfer.

Different methods and training strategies are convenient to compose and add. We hope that this toolbox could help people in research use.

## Features
-  [x] Implementations of sota inversion methods.
-  [x] Uniform training/evaluation/inference/editing process.
-  [x] Modular and flexible configuration. Easy to set options by config file (yaml) or command in every use.
-  [x] Training features
    -  [ ] Distributed training.
    -  [x] Weight & bias (wandb).
    -  [x] Automatically resume training.
-  [ ] More methods.

## Citation
If you use this toolbox for your research, please cite our repo.
```
@misc{cao2022ganinverter,
  author       = {Pu Cao and Dongxu Liu and Lu Yang and Qing Song},
  title        = {GAN Inverter},
  howpublished = {\url{https://github.com/caopulan/GANInverter}},
  year         = {2022},
}
```
