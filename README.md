![GAN Inverter](./docs/gan_inverter.jpeg)

GAN Inverter is a GAN inversion toolbox based on PyTorch library. 

We collect SOTA inversion methods and construct a uniform pipeline with more features. Different methods and training strategies are convenient to compose and add. We hope that this toolbox could help people in every use.

## Table of Contents
  * [Features](#features)
  * [Recent Updates](#recent-updates)
  * [Model Zoo](#model-zoo)
  * [Benchmark](#benchmark)
  * [Unified Pipeline](#unified-pipeline)
  * [Installation](#installation)
  * [Getting Start](#getting-start)
    + [Preparing Data and Generator](#preparing-data-and-generator)
    + [Training](#training)
    + [Inference](#inference)
    + [Editing](#editing)
    + [Evaluation](#evaluation)
  * [Citation](#citation)

## Features
-  [x] Implementations of sota inversion methods.
-  [x] Unified training/evaluation/inference/editing process.
-  [x] Modular and flexible configuration. Easy to set options by config file (yaml) or command in every use.
-  [x] Additional training features.
    -  [x] Distributed training.
    -  [x] Weight & bias (wandb).
    -  [x] Automatically resume training.
-  [x] Evaluation system and inversion benchmark.
-  [x] More editing methods.

## Recent Updates

**We are working for supporting more methods' inference and conducting the benchmark.**

**`2023.05`**: Benchmark v1.0 is released. We now support: pSp, e4e, LSAP, ReStyle, HyperStyle, PTI, SAM, HFGI, DHR (some of them only for inference). Have fun!

**`2023.02`**: SAM is supported.

**`2023.02`**: V1.1. Re-organized codes: methods' class, inference pipeline. Add our new work DHR ["What Decreases Editing Capability? Domain-Specific Hybrid Refinement for Improved GAN Inversion"](https://arxiv.org/abs/2301.12141).

**`2022.11`**: Add more optimizers and PTI is supported now.

**`2022.10`**: GAN Inverter v1.0 is released. Support methods: pSp, e4e, LSAP.

**`2022.09`**: LSAP is published on [arxiv](https://arxiv.org/abs/2209.12746).

## Model Zoo
Although previous works use "encoder-based", "optimization-based" and "hybrid method" to categorize inversion methods, this set of division criteria is no longer appropriate at present.
According to the purpose of methods, we divide the inversion process into two steps: **Image Embedding** and **Result Refinement**:

- **Image Embedding** aims to embed images into latent code by encoder or optimization.
- **Result Refinement** aims to refine the initial inversion and editing results from the first step by various strategies (e.g., adjusting generator weight or intermediate feature).
### 1. Image Embedding Methods
|      |      Method       | Type |                           Repo                           |                                                  Paper                                                  |    Source    |
| :--: |:-----------------:|:----:|:--------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|:------------:|
|   :ballot_box_with_check:   |        pSp        |  E   |  [code](https://github.com/eladrich/pixel2style2pixel)   |                                [paper](https://arxiv.org/abs/2008.00951)                                |   CVPR2021   |
|   :ballot_box_with_check:   |        e4e        |  E   |    [code](https://github.com/omertov/encoder4editing)    |                                [paper](https://arxiv.org/abs/2102.02766)                                | SIGGRAPH2021 |
|   :ballot_box_with_check:   |       LSAP        |  E   |                  [code](./configs/lsap)                  |                                [paper](https://arxiv.org/abs/2209.12746)                                |  Arxiv2022   |
|   :ballot_box_with_check:   |      ReStyle      |  E   |                [code](./configs/restyle)                 |                               [paper](https://arxiv.org/abs/2104.02699v2)                               |   ICCV2021   |
|   :white_medium_square:   |      E2Style      |  E   |       [code](https://github.com/wty-ustc/e2style)        |                     [paper](https://wty-ustc.github.io/inversion/paper/E2Style.pdf)                     |   TIP2022    |
|   :white_medium_square:   | Style Transformer |  E   | [code](https://github.com/sapphire497/style-transformer) |                                [paper](https://arxiv.org/abs/2203.07932)                                |   CVPR2022   |
|   :ballot_box_with_check:   | StyleGAN2(LPIPS)  |  O   | [code](https://github.com/rosinality/stylegan2-pytorch)  |                                [paper](http://arxiv.org/abs/1912.04958)                                 |   CVPR2020   |

Note: ```E```/```O```/```H``` means encoder-based and optimization-based methods.

### 2. Result Refinement Methods

|      |      Method      |                        Repo                        |                   Paper                    |  Source  |
| :--: | :--------------: | :------------------------------------------------: | :----------------------------------------: | :------: |
|   :ballot_box_with_check:   |    HyperStyle    | [code](https://github.com/yuval-alaluf/hyperstyle) | [paper](https://arxiv.org/abs/2111.15666)  | CVPR2022 |
|   :ballot_box_with_check:   |       HFGI       |    [code](https://github.com/Tengfei-Wang/HFGI)    | [paper](https://arxiv.org/pdf/2109.06590)  | CVPR2022 |
|   :ballot_box_with_check:   |       SAM       | [code](https://github.com/adobe-research/sam_inversion) | [paper](https://arxiv.org/abs/2206.08357) | CVPR2022 |
|   :ballot_box_with_check:   |       PTI        |     [code](https://github.com/danielroich/PTI)     | [paper](https://arxiv.org/abs/2106.05744) | TOG2022  |
|   :white_medium_square:   | FeatureStyleEncoder | [code](https://github.com/InterDigitalInc/FeatureStyleEncoder) | [paper](https://link.springer.com/chapter/10.1007/978-3-031-19784-0_34) |  ECCV2022  |
| :ballot_box_with_check: | Domain-Specific Hybrid Refinement (DHR) | [code](./configs/dhr) | [paper](https://arxiv.org/abs/2301.12141) | Arxiv2023 |

### 3. Editing Methods

|      |    Method    |                       Repo                       |                   Paper                   |   Source    |
| :--: | :----------: | :----------------------------------------------: | :---------------------------------------: | :---------: |
|  :ballot_box_with_check:    | InterFaceGAN | [code](https://github.com/genforce/interfacegan) | [paper](https://arxiv.org/pdf/1907.10786) |  CVPR2020   |
|  :ballot_box_with_check:    |   GANSpace   |   [code](https://github.com/harskish/ganspace)   | [paper](https://arxiv.org/abs/2004.02546) | NeurIPS2020 |
|   :white_medium_square:   |  StyleClip   | [code](https://github.com/orpatashnik/StyleCLIP) | [paper](https://arxiv.org/abs/2103.17249) |  ICCV2021   |

## Benchmark 

As evaluation settings are different in previous inversion works, we conduct a benchmark to better evaluate inversion methods based on our unified pipeline. See [Evaluation](#evaluation) for more details.

**Evaluation Settings:**

- **Dataset:** CelebA-HQ test split (2,824) images;
- **ID:** identity similarity measured by [face recognition model]();
- **LPIPS version:** VGG;
- Images are generated and converted to uint8 to evaluate, except for ID and FID, which are evaluated on saved images (png format).
- See `scripts/test.py` for more details.

|  Refinement   |    Embedding    | PSNR $\uparrow$ | MSE $\downarrow$ | LPIPS $\downarrow$ | ID $\uparrow$ | FID $\downarrow$ | Latency |
|:-------------:|:---------------:|:---------------:|:----------------:|:------------------:|:-------------:|:----------------:|:-------:|
|       -       | Optimization-W  |     15.8460     |      0.0670      |       0.1967       |     0.32      |     25.7447      |    -    |
|       -       | Optimization-W+ |     20.5940     |      0.0242      |       0.1092       |     0.77      |     17.0812      |    -    |
|       -       |       pSp       |     18.0348     |      0.0345      |       0.1591       |     0.56      |     25.2540      |  41ms   |
|       -       |       e4e       |     16.6616     |      0.0472      |       0.1974       |     0.50      |     28.4952      |  42ms   |
|       -       |      LSAP       |     17.4958     |      0.0391      |       0.1765       |     0.53      |     29.3118      |  42ms   |
|       -       |   ReStyle-e4e   |     17.0903     |      0.0428      |       0.1904       |     0.51      |     25.5141      |  150ms  |
|  HyperStyle   |    W-encoder    |     20.0864     |      0.0219      |       0.0985       |     0.74      |     21.6660      |  162ms  |
| PTI $\dagger$ |     W-pivot     |     24.6131     |      0.0082      |       0.0817       |     0.85      |     14.2792      |   59s   |
|     HFGI      |       e4e       |     20.1402     |      0.0210      |       0.1166       |     0.68      |     16.0659      |  67ms   |
|      SAM      |       e4e       |     20.5933     |      0.0193      |       0.1442       |     0.57      |     17.3631      |   67s   |
|      SAM      |      LSAP       |     21.6179     |      0.0152      |       0.1205       |     0.60      |     15.2710      |   67s   |
|      DHR      |       e4e       |     28.1661     |      0.0035      |       0.0438       |     0.87      |      5.9960      |   12s   |
|      DHR      |      LSAP       |     28.2786     |      0.0034      |       0.0422       |     0.88      |      6.0594      |   12s   |

**Note:** 
- $\dagger$: we don't apply regularization in PTI, following [issue](https://github.com/danielroich/PTI/blob/5f4c05726caa908a46537c5aaca6ebc8ea34201e/configs/hyperparameters.py#L9) and [official config]()
- *We recommend using this benchmark and evaluation settings to evaluate inversion methods in future work.*
- Latency is tested on a single RTX3090 of batchsize 1 (TODO).
- The results may be inconsistent with the reported results in our paper because of different implementations.
- There are very small numerical differences in some values during multiple measurements.

## Unified Pipeline

### Configs

We conduct a unified config system in train/inference/test/edit. All options are saved in the config file, which can be conveniently determined for any use. 

We define all options in ```options```.  And ```options/base_options.py``` contains communal options in every phase. 

### Two-Stage Inference Pipeline

<img src="./docs/inference_pipeline.png" alt="inference_pipeline" style="zoom: 15%;" />

We follow two-stage inference in this repository. The base inference class `TwoStageInference` is defined in ```./inference/two_stage_inference.py```. It follows **image embedding -> result refinement** pipeline.

This uniform inversion process can easily combine two methods. **Users can try any combination of methods, not limited to those employed by the original authors.** For example, GANInverter makes it possible to connect ReStyle with HyperStyle by ```--embed_mode restyle --refine_mode hyperstyle``` or PTI + e4e by `--embed_mode e4e --refine_mode pti`. 

You can run any method combination by setting their config files now. See [Inference](#inference) for more details.

For example:

- e4e: `--configs configs/e4e/e4e_ffhq_r50.yaml`
- PTI + e4e: ` --configs configs/e4e/e4e_ffhq_r50.yaml configs/pti/pti.yaml`
- DHR + saved latent codes: `--embed_mode code --refine_mode dhr --code_path /path/to/code/xxxx.pt`

### Model checkpoint
You can load a checkpoint by two ways:
- ```--checkpoint_path xxxx.pt```: manually set checkpoint path to load. Although model architecture is slightly different from previous repository (e.g., pSp, e4e), the weight will be automatically converted to fit our architecture. You can use their original weight file.
- ```--auto_resume True```: automatically load ```{exp_dir}/checkpoints/last.pt``` in training phase or ```{exp_dir}/checkpoints/best_model.pt``` in the other phase.
### Dataset
- ```--train_dataset_path```, ```--batch_size```, ```--workers``` are used only in training.
- ```--test_dataset_path```, ```--test_batch_size```, ```--test_workers``` are default test/inference options in every use.


## Installation
Please refer to [Installation Instructions](docs/install.md) for the details of installation.

## Getting Start
### Preparing Data and Generator weight
Please refer to [Dataset Instructions](docs/dataset.md) for the details of datasets.

### Download pre-trained model weights
Using `download_weight.sh` to easily download pre-trained weights of pSp, e4e, ReStyle, PTI, HFGI, SAM, DHR.
```bash
# To download all models (including all methods and auxiliary models).
sh download_weight.sh all
# To download a specific model
sh download_weight.sh e4e
sh download_weight.sh lsap
# To download auxiliary training weights (e.g., face recognition model for id_loss)
sh download_weight.sh train
# To download auxiliary evaulation weights (e.g., face recognition model for id_loss)
sh download_weight.sh evaluation
```

### Training

#### 1. Train encoder
**Example**: Train LSAP on FFHQ

##### **(1) Single-card training**

```bash
python scripts/train.py -c configs/lsap/lsap_ffhq_r50.yaml
```
##### **(2) Distributed training on 8 Cards**

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 -c configs/lsap/lsap_ffhq_r50.yaml --gpu_num 8
```

**Notes:**
- set `--auto_resume True` for automatically resume.
- Batch size means total size of all gpus. It must be a multiple of gpu num.
- In our experiments, distributed training with batch size of 8 may much slower or accelerate marginally. For example, one iteration of e4e cost 50 sec on both one or two cards. However, distributed training can amplify the total batch size (batch size 8 cost 21G gpu memory) and may achieve fast convergence by large batch size and learning rate. If batch size increased to 16 on two cards (8 sample per card), cost per iteration only slightly increase (from 50 to <60 sec). We are glad to receive any suggestions to improve performance of distributed training.

#### 2. [TODO] Train refinement model based on encoder 

**Example: Train HFGI with LSAP on FFHQ.** 

### Inference

You can infer images by:

``````
python scripts/infer.py -c /path/to/config1 /path/to/config2
``````

or

```
python scripts/infer.py \
	--embed_mode [embed_mode] \
	--refine_mode [refine_mode] \
	--test_dataset_path [/path/to/dataset]\
	--output_dir [/path/to/output] \
	--save_code [true/false] \
	--checkpoint_path [/path/to/checkpoint]
```

- ```--save_code```: whether to save latent code.
- ```--test_dataset_path```: image file or folder.
- ```--output_dir```: path to save inversion results. Inverse images will be saved in ```{output_dir}/inversion/``` and latent codes will be saved in ```{output_dir}/code/```. If not set, use ```{exp_dir}/inference/``` by default.
- ```--checkpoint_path```: model weight. 

**Example1: LSAP on CelebA-HQ.** 

```bash
python scripts/infer.py -c configs/lsap/lsap_ffhq_r50.yaml
```
**Example2: Optimization on CelebA-HQ.** 

```bash
python scripts/infer.py -c configs/optim/optim_celeba-hq.yaml
```
- ```--stylegan_weights```: whether to save latent code.

**Example3: PTI+e4e**

```
python scripts/infer.py -c configs/e4e/e4e_ffhq_r50.yaml configs/pti/pti.yaml
```

### Editing

We have three ``embed_mode`` in editing: ```encoder```, ```optim```, and ```code```.
#### (1) Edit with encoder or optimization
```bash
python scripts/edit.py -c configs/lsap/lsap_ffhq_r50.yaml --edit_mode interfacegan --edit_path editing/interfacegan_directions/age.pt --edit_factor 1.0
```
#### (2) Edit with inverse codes
If you have inferred images first and saved the latent codes, you can edit these latent codes without inversion. We recommend "inference->edit" pipeline since editing with various attributes and factors will not cost extra inversion time.
```bash
python scripts/edit.py -c configs/optim/optim_celeba-hq.yaml --embed_mode code --test_dataset_path /path/to/latent/codes/ --edit_mode interfacegan --edit_path editing/interfacegan_directions/age.pt --edit_factor 1.0
```

### Evaluation
#### (1) PSNR, MSE, and LPIPS
These three metrics can be evaluated during.
```bash
python scripts/edit.py -c configs/e4e/e4e_ffhq_r50.yaml
```
#### (2) FID
You need saving inference results to evaluate fid by pytorch-fid.
```bash
python -m pytorch-fid /path/to/dataset/ /path/to/inference/results/
```
#### (3) ID
You need saving inference results to evaluate id similarity by face recognition model.
```bash
python scripts/calc_id_loss.py --gt_path /path/to/dataset/ --data_path /path/to/inference/results/
```


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

```
@article{cao2022lsap,
  title={LSAP: Rethinking Inversion Fidelity, Perception and Editability in GAN Latent Space},
  author={Cao, Pu and Yang, Lu and Liu, Dongxv and Liu, Zhiwei and Wang, Wenguan and Li, Shan and Song, Qing},
  journal={arXiv preprint arXiv:2209.12746},
  year={2022}
}
```
