# Dataset Instruction

|   Domain    |   Dataset    | Generator | Height | Width | Comment |
| :---------: | :----------: | :-------: | :----: | :---: | :-----: |
|    Face     |     [FFHQ](#ffhq)     | [StyleGAN2*](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl)<br />[convert](https://drive.google.com/file/d/1fgehC3QTtEayc_AFdsCx-UkxU8d1d9zW/view?usp=sharing) |  1024  | 1024  |         |
|    Face     |    [CelebA](#celeba-hq)    | - |  1024  | 1024  | evaluation |
|     Cat     |   [AFHQ Cat](#afhq)   | [StyleGAN2-ADA](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl) | 512 | 512 |         |
|     Cat     |   [LSUN Cat](#lsun)   | [StyleGAN2*](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-cat-config-f.pkll)<br />[convert](https://drive.google.com/file/d/1s6x2BApGb0Sfp_hQyi59yM0IES5qYlBM/view?usp=sharing) | 256 | 256 |         |
|     Dog     |   [AFHQ Dog](#afhq)   | [StyleGAN2-ADA](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl) | 512 | 512 |         |
| Wild Animal |  [AFHQ Wild](#afhq)   | [StyleGAN2-ADA](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/#:~:text=41%3A12%20AM-,afhqwild.pkl,-363959591) | 512 | 512 |         |
|    Horse    |  [LSUN Horse](#lsun)  | [StyleGAN2*](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-horse-config-f.pkl)<br />convert | 256 | 256 |         |
|     Car     | [Stanford Car](#stanford-car) | - |        |       | train inversion |
|     Car     |   [LSUN Car](#lsun)   | [StyleGAN2*](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-car-config-f.pkl)<br />[convert](https://drive.google.com/file/d/1d5ATre9K1cVo9m6WCzjrONOPsMdhrHL3/view?usp=sharing) | 384 | 512 | Crop    |
|   Church    | [LSUN Church](#lsun)  | [StyleGAN2*](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-church-config-f.pkl)<br />[convert](https://drive.google.com/file/d/1TWeD0I2zfB53LAKoQ16hTnblZ2AhspJu/view?usp=sharing) | 256 | 256 |         |

* ```*```: weight is from the original [StyleGAN2]([NVlabs/stylegan2: StyleGAN2 - Official TensorFlow Implementation (github.com)](https://github.com/NVlabs/stylegan2)) (TensorFlow-based), which needs to convert by [script](scripts/convert_weight.py). We also provide the converted weights, which are converted by [this implementation](https://github.com/dvschultz/stylegan2-ada-pytorch/blob/main/export_weights.py). 


## Face Dataset
We use FFHQ (70,000 images) for training and CelebA-HQ test dataset (2824 images) for testing.
### FFHQ
Download script can be found in [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset).

By default, the image path follows: ```data/FFHQ/xxxxx.png```.

### CelebA-HQ Test
CelebA-HQ is a subset of CelebA dataset, we share the test split (2824 images) on drive.

By default, the image path follows: ```data/CelebA-HQ/test/xxxxxx.jpg```.

## AFHQ (Animal)
AFHQ consists of three categories: cat, dog, and wild. There are two versions (i.e., AFHQ and AFHQv2), and we use AFHQ by default. 

TODO

## LSUN (Scene and Object)

TODO