
### 1. Environment setup
```bash
conda env create -f environment.yaml
conda activate eqvae_train
pip install packaging==21.3
pip install 'torchmetrics<0.8'
pip install transformers==4.10.2
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install Pillow==9.5.0
```


### 2. Download SD-VAE
To download the SD-VAE from the official LDM repository run:


```bash
bash download_sdvae.sh
```



### 3. Dataset

#### Dataset download

Currently, we provide experiments for [OpenImages](https://storage.googleapis.com/openimages/web/index.html). After downloading modify paths of train_dir, val_dir, dataset_name in the [cofig file](configs/eqvae_config.yaml)









### 4. Training

To run EQ-VAE regularization on 8 GPUs:

```bash
bash imagenet256_EQVAE_pytorch_ddp.sh
```

All model, data, validation, logging, distributed, and checkpoint options live in `configs/eqvae_config.yaml`.
For one GPU, run `bash imagenet256_EQVAE_pytorch_single_gpu.sh`.

- `anisotropic`: If `True` will do anisotropic scaling 
- `uniform_sample_scale`: If `True` will sample scale factors uniformly from `[0.25, 1)` if set to `False` will randomly choose from scales from `{0.25, 0.5, 0.75}`.
- `p_prior`: Probability of the latent equivariance branch.
- `p_prior_s`: Probability of low-resolution prior preservation inside the image branch.
  