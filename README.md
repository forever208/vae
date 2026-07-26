# EQ-VAE training

This repository contains the self-contained PyTorch implementation for training
and evaluating the first-stage EQ-VAE. It supports single-GPU execution and
multi-GPU DistributedDataParallel (DDP) training.

### 1. Environment setup

Run the setup command from the repository root because `environment.yaml`
installs the local project in editable mode:

```bash
conda env create -f environment.yaml
conda activate eqvae_train
```

The small `taming-transformers` subset needed by LPIPS and PatchGAN is already
integrated into `losses/` and `models/`; no nested Git checkout is required.

### 2. Download SD-VAE

Download the SD-VAE checkpoint from the official LDM repository:

```bash
bash download_sdvae.sh
```

The script places the checkpoint at `pretrained_models/model.ckpt`. LPIPS
downloads its VGG weights to `pretrained_models/vgg.pth` automatically on the
first run if they are not already present.

### 3. Dataset

The data loader supports both
[ImageNet](https://www.image-net.org/) and
[OpenImages](https://storage.googleapis.com/openimages/web/index.html).

- `dataset_name: imagenet` recursively scans class subdirectories.
- `dataset_name: openimages` scans images directly under the configured
  directory.
- Supported image extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`.

After downloading the dataset, update the following entries in
`configs/eqvae_config.yaml`:

```yaml
data:
  train:
    params:
      train_dir: /path/to/train
      dataset_name: imagenet
  validation:
    params:
      val_dir: /path/to/validation
      dataset_name: imagenet
```

The default pipeline produces a normalized full-resolution `image` and a
downsampled `LR_image`. The current EQ-VAE training path consumes `image`;
`LR_image` remains available for degradation-based experiments. Available
degradations include PIL/OpenCV interpolation, `bsrgan`, and `bsrgan_light`.

### 4. Training

#### 4.1 Configure the run

Before launching training, review `configs/eqvae_config.yaml` and set:

- `data.train.params.train_dir` and
  `data.validation.params.val_dir`
- `data.*.params.dataset_name` to `imagenet` or `openimages`
- `pytorch_training.checkpoint.output_dir`
- `pytorch_training.checkpoint.resume_path`

`stage1_vae_trainer.py` is the main entry point. It loads the complete YAML
configuration, initializes the distributed context, builds the model and its
two optimizers, restores a checkpoint, creates the data loaders, and starts
the training loop.

#### 4.2 Single-GPU training

From the repository root:

```bash
conda activate eqvae_train
CUDA_VISIBLE_DEVICES=0 python stage1_vae_trainer.py \
  --config configs/eqvae_config.yaml
```

#### 4.3 Multi-GPU training

The provided launchers start eight-GPU local DDP jobs:

```bash
bash imagenet256_EQVAE.sh
bash imagenet256_DSM.sh
```

Before using it on another machine, update its Conda and cache paths as needed.
The equivalent portable command is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 --master_port=29507 \
  stage1_vae_trainer.py --config configs/eqvae_config.yaml
```

`torchrun` provides `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`; the trainer uses
these values to initialize NCCL, select one CUDA device per process, and add a
`DistributedSampler` to each data loader.

#### 4.4 Optimization and step semantics

Each batch performs two passes:

1. Update the autoencoder using reconstruction, LPIPS, KL, and adaptive
   adversarial losses.
2. Update the PatchGAN discriminator.

`local_step` increases once per batch. `global_step` increases after each
optimizer update, so with `accumulate_grad_batches: 1` it normally increases
twice per batch. Logging and validation intervals are expressed in
`local_step`.

When `scale_learning_rate: true`, the effective learning rate is:

```text
base_learning_rate × batch_size × world_size × accumulate_grad_batches
```

With the default configuration this is `9e-5` on one GPU and `7.2e-4` on
eight GPUs. The current trainer executes in FP32; the `precision` configuration
field is reserved and does not currently enable automatic mixed precision.

#### 4.5 DSM-specific options

- `p_prior`: probability of applying DSM block-DCT low-pass regularization to
  both the input target and sampled latent. A value of `0.0` disables DSM.
- `p_prior_s`: probability of resizing the input in the normal image branch.
  A value of `0.0` disables this low-resolution prior.
- `blk_sz`: DCT block size. It must divide both image and latent spatial
  dimensions; the ImageNet 256 configuration uses `8`.
- A DSM step samples its frequency mask from `{8, 10, 12}`.
- `lossconfig.params.kl_weight`, `disc_weight`, and `disc_start` control the
  KL term, adversarial weight, and discriminator start step.

#### 4.6 Validation, outputs, and checkpoints

Validation runs every
`pytorch_training.validation.interval_local_steps` batches and computes FID,
PSNR, LPIPS, and SSIM over the validation set. Rank 0 writes all persistent
outputs:

```text
<output_dir>/
├── resolved_config.yaml
├── eval.log
├── <local_step>.png
└── checkpoints/
    ├── last.pt
    ├── topk.json
    └── step_<global_step>.pt
```

The checkpoint manager keeps `save_top_k` checkpoints according to
`checkpoint.monitor` and `checkpoint.mode`, in addition to `last.pt` when
`save_last` is enabled. The eight-GPU shell launcher also writes timestamped
stdout and stderr logs under `<output_dir>/logs/`.

#### 4.7 Starting from pretrained weights or resuming

To initialize from a weights-only checkpoint, disable optimizer restoration:

```yaml
pytorch_training:
  checkpoint:
    resume_path: pretrained_models/model.ckpt
    resume_optimizer: false
    strict_model_load: false
```

To continue a previous EQ-VAE run, restore `last.pt` and both optimizer states:

```yaml
pytorch_training:
  checkpoint:
    resume_path: /path/to/output/checkpoints/last.pt
    resume_optimizer: true
    strict_model_load: false
```

Set `resume_path` to `null` to train without loading a checkpoint. Checkpoints
restore the model, optimizer states (when requested), `global_step`, epoch, and
Top-K metadata. The validation `local_step` schedule restarts from zero after
resume.

### 5. Repository structure and implementation details

```text
eqvae/
├── configs/
│   ├── eqvae_config_imgnet256.yaml
│   └── dsm_config_imgnet256.yaml
├── core/
│   └── config.py               # Dynamic target-based object construction
├── data/
│   ├── datasets.py             # ImageNet/OpenImages datasets
│   └── degradation/            # BSRGAN and interpolation degradations
├── models/
│   ├── vae.py                  # AutoencoderKL and DSM train branch
│   ├── backbone.py             # Encoder and decoder
│   ├── discriminator.py        # PatchGAN discriminator
│   └── distributions.py        # Diagonal Gaussian posterior
├── losses/
│   ├── vae_loss.py             # Reconstruction, LPIPS, KL, and GAN objective
│   ├── lpips.py                # Perceptual feature loss
│   └── gan.py                  # Adversarial loss helpers
├── training/
│   ├── builders.py             # Model, optimizer, DDP, and DataLoader setup
│   ├── loop.py                 # Dual-optimizer epoch loop
│   ├── distributed.py          # Process group, device, and seed handling
│   ├── metrics.py              # In-training FID/PSNR/LPIPS/SSIM validation
│   ├── logging.py              # Progress, JSONL, eval, and image logging
│   └── checkpoint.py           # Load/save and Top-K retention
├── evaluation/                 # Standalone offline evaluation utilities
├── tools/                      # DDP smoke and checkpoint comparison tools
├── stage1_vae_trainer.py       # Primary training entry point
├── imagenet256_EQVAE.sh
└── imagenet256_DSM.sh          # DSM eight-GPU launcher
```

The main execution flow is:

```text
stage1_vae_trainer.py
  → DistributedContext and reproducibility setup
  → builders.py: model, two Adam optimizers, optional DDP wrapper
  → checkpoint.py: optional model/optimizer restoration
  → builders.py: train and validation data loaders
  → loop.py: autoencoder pass + discriminator pass
  → metrics.py: periodic distributed validation
  → logging.py and checkpoint.py: rank-0 outputs
```

`models/vae.py` chooses between two training branches. The image branch
reconstructs the input directly, with optional low-resolution prior sampling.
The DSM branch applies the same randomly sampled block-DCT low-pass mask to
the input target and sampled latent before decoding. Both branches share the
same `LPIPSWithDiscriminator` objective, while KL is computed from the
unfiltered posterior.

The standalone scripts under `evaluation/` do not participate in the
in-training validation path. In particular, `evaluation/vis_latent.py`
requires the optional `diffusers` package, which is not part of the core
training environment.
  