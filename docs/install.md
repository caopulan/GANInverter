# Installation Instruction

- python 3.8
- Pytorch 1.12.1 with CUDA11.6
### 1. Create the conda environment
```bash
conda create -n gan_inverter python=3.8
conda activate gan_inverter
```
### 2. Install Pytorch

Pytorch 1.12.1 with CUDA 11.6 is tested. Please install pytorch by [official website](https://pytorch.org/get-started/locally/).
### 3. Install Requirements
```bash
pip3 install -r requirements.txt
```

### Note
- If you meet any error during StyleGAN opt building, please search issues in official StyleGAN series repositories first.
- We recommend CUDA 11.x and Pytorch version > 1.9.
- We have not tested cpu environment.