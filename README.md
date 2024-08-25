# Image Neural Field Diffusion Models

![infd](https://github.com/user-attachments/assets/e8296750-6ec0-4917-8eb5-7dedd6c85dbb)

Official implementation of the paper:

[**Image Neural Field Diffusion Models**](https://arxiv.org/abs/2406.07480)
<br>
Yinbo Chen, Oliver Wang, Richard Zhang, Eli Shechtman, Xiaolong Wang, Michael Gharbi
<br>
CVPR 2024 (Highlight)

Contact yinboc96@gmail.com for any issues about the code.

## Environment
```
conda create -n infd python=3.8 -y
conda activate infd
pip install -r requirements.txt
```

## Training

Below shows an example for training on FFHQ-1024 with 8 GPUs.

Download the FFHQ dataset ([images1024x1024.zip](https://drive.google.com/drive/folders/1WocxvZ4GEZ1DI8dOz30aSj2zT6pkATYS)). Unzip it and put the image folder as `load/ffhq/ffhq_1024`.

To visualize with wandb, complete information in `wandb.yaml` and append `-w` in running commands.

To train for the FFHQ-6K-Mix setting, append `-mix6000` to the yaml config names. 

### 1. Autoencoding stage
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc-per-node=8 run.py --cfg cfgs/ae_ffhq.yaml
```

### 2. Latent diffusion stage

First resize the images for faster loading:
```
python resize_images.py --input load/ffhq/ffhq_1024 --output load/ffhq/ffhq_lanczos256
```

Then run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc-per-node=8 run.py --cfg cfgs/dm_ffhq.yaml
```

### Custom Datasets

To train on custom datasets, use `ae_custom.yaml`, `dm_custom.yaml` as cfg and replace root_path in configs with path to the image folder.

## Evaluation

### 1. Generate samples

Can use a single or multiple GPUs. For example, with 2 GPUs:
```
CUDA_VISIBLE_DEVICES=0,1 python gen_samples.py --model save/dm_ffhq/last-model.pth --n-samples 50000 --batch-size 32 -o save/gen_samples --output-sizes 1024
```

By default it uses the sampler defined in the model (200 DDIM steps, eta=1, following LDM).

### 2. Evaluate patch FID

```
CUDA_VISIBLE_DEVICES=0 python eval_pfid.py --input1 load/ffhq/ffhq_1024 --input2 save/gen_samples/1024
```

## Citation
```
@inproceedings{chen2024image,
  title={Image Neural Field Diffusion Models},
  author={Chen, Yinbo and Wang, Oliver and Zhang, Richard and Shechtman, Eli and Wang, Xiaolong and Gharbi, Michael},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8007--8017},
  year={2024}
}
```