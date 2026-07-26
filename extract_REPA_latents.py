"""Extract posterior moments from an EQVAE VAE for REPA training."""

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from models.vae import AutoencoderKL


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs/sdvae_config_imgnet256.yaml"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "exps/imagenet256_SDVAE_ft/checkpoints/4000.ckpt"
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def center_crop(width, height, image):
    image = np.asarray(image)
    crop = min(image.shape[:2])
    image = image[
        (image.shape[0] - crop) // 2:(image.shape[0] + crop) // 2,
        (image.shape[1] - crop) // 2:(image.shape[1] + crop) // 2,
    ]
    image = PIL.Image.fromarray(image, "RGB")
    return np.asarray(
        image.resize((width, height), PIL.Image.Resampling.LANCZOS)
    )


def center_crop_wide(width, height, image):
    image = np.asarray(image)
    crop_height = int(round(width * image.shape[0] / image.shape[1]))
    if image.shape[1] < width or crop_height < height:
        raise ValueError(
            f"Image shape {image.shape[:2]} is too small for a "
            f"{width}x{height} center-crop-wide transform"
        )
    image = image[
        (image.shape[0] - crop_height) // 2:
        (image.shape[0] + crop_height) // 2
    ]
    image = PIL.Image.fromarray(image, "RGB")
    image = np.asarray(
        image.resize((width, height), PIL.Image.Resampling.LANCZOS)
    )
    canvas = np.zeros((width, width, 3), dtype=np.uint8)
    canvas[(width - height) // 2:(width + height) // 2, :] = image
    return canvas


def center_crop_arr(image, image_size):
    """Center-crop implementation used by ADM and REPA."""
    while min(*image.size) >= 2 * image_size:
        image = image.resize(
            tuple(size // 2 for size in image.size),
            resample=PIL.Image.Resampling.BOX,
        )
    scale = image_size / min(*image.size)
    image = image.resize(
        tuple(round(size * scale) for size in image.size),
        resample=PIL.Image.Resampling.BICUBIC,
    )
    array = np.asarray(image)
    crop_y = (array.shape[0] - image_size) // 2
    crop_x = (array.shape[1] - image_size) // 2
    return PIL.Image.fromarray(
        array[crop_y:crop_y + image_size, crop_x:crop_x + image_size]
    )


def build_transform(name, resolution):
    transform_fns = {
        "center-crop": lambda image: center_crop(resolution, resolution, image),
        "center-crop-wide": lambda image: center_crop_wide(
            resolution, resolution, image
        ),
        "center-crop-dhariwal": lambda image: center_crop_arr(image, resolution),
    }
    return transforms.Compose(
        [
            transforms.Lambda(transform_fns[name]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                inplace=True,
            ),
        ]
    )


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index


class DistributedSliceSampler(Sampler):
    """Shard indices among ranks without DistributedSampler padding."""

    def __init__(self, dataset_size, rank, world_size):
        self.indices = range(rank, dataset_size, world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _resolve_project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_encoder(config_path, checkpoint_path, device):
    config_path = _resolve_project_path(config_path)
    checkpoint_path = _resolve_project_path(checkpoint_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    config = OmegaConf.load(config_path)
    model = AutoencoderKL(**config.model.params)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError(
            f"Checkpoint {checkpoint_path} has no 'model' or 'state_dict' entry"
        )
    if state and next(iter(state)).startswith("module."):
        state = {
            key.removeprefix("module."): value for key, value in state.items()
        }
    incompatible = model.load_state_dict(state, strict=False)
    required_missing = [
        key
        for key in incompatible.missing_keys
        if key.startswith(("encoder.", "quant_conv."))
    ]
    if required_missing:
        preview = ", ".join(required_missing[:10])
        raise RuntimeError(f"Checkpoint is missing encoder weights: {preview}")

    # Only the encoder and quantization projection are needed for extraction.
    model.decoder = None
    model.post_quant_conv = None
    model.loss = None
    return model.eval().requires_grad_(False).to(device)


def _output_names(index):
    index_string = f"{index:08d}"
    directory = index_string[:5]
    image_name = f"{directory}/img{index_string}.png"
    latent_name = f"{directory}/img-mean-std-{index_string}.npy"
    return image_name, latent_name


def _save_batch(dest, dest_images, entries, save_images):
    worker_count = max(1, min(32, (os.cpu_count() or 1) * 2))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for latent_name, latent, image_name, image in entries:
            latent_path = dest / latent_name
            latent_path.parent.mkdir(parents=True, exist_ok=True)
            futures.append(executor.submit(np.save, latent_path, latent))
            if save_images:
                image_path = dest_images / image_name
                image_path.parent.mkdir(parents=True, exist_ok=True)
                futures.append(executor.submit(image.save, image_path))
        for future in futures:
            future.result()


@torch.inference_mode()
def _compute_latent_statistics(model, loader, device, num_stat_samples):
    """Compute global scalar mean/std over sampled encoder latents."""
    if num_stat_samples <= 0:
        raise ValueError("num_stat_samples must be greater than zero")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dataset_size = len(loader.dataset)
    total_target = min(num_stat_samples, dataset_size)
    local_target = max(
        0, (total_target + world_size - 1 - rank) // world_size
    )

    sample_count = 0
    value_sum = torch.zeros((), dtype=torch.float64, device=device)
    square_sum = torch.zeros((), dtype=torch.float64, device=device)
    value_count = torch.zeros((), dtype=torch.int64, device=device)

    for images, _, _ in tqdm(
        loader,
        total=math.ceil(local_target / loader.batch_size),
        desc="Computing latent statistics",
        disable=rank != 0,
    ):
        if sample_count >= local_target:
            break
        images = images.to(device, non_blocking=True)
        posterior = model.encode(images)
        take = min(local_target - sample_count, images.shape[0])
        latents = posterior.sample()[:take].to(torch.float64).reshape(-1)
        value_sum += latents.sum()
        square_sum += latents.square().sum()
        value_count += latents.numel()
        sample_count += take

    reduced_sample_count = torch.tensor(
        sample_count, dtype=torch.int64, device=device
    )
    for statistic in (
        value_sum,
        square_sum,
        value_count,
        reduced_sample_count,
    ):
        dist.all_reduce(statistic, op=dist.ReduceOp.SUM)

    if value_count.item() == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "scaling_factor": float("nan"),
            "num_samples": 0,
        }

    mean = value_sum / value_count
    variance = torch.clamp(square_sum / value_count - mean.square(), min=0.0)
    std = torch.sqrt(variance)
    std_value = std.item()
    return {
        "mean": mean.item(),
        "std": std_value,
        "scaling_factor": 1.0 / std_value if std_value > 0 else float("nan"),
        "num_samples": reduced_sample_count.item(),
    }


def _initialize_runtime(args):
    if not torch.cuda.is_available():
        raise RuntimeError("VAE encoding requires at least one CUDA GPU")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed * world_size + rank)

    source = Path(args.source).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")

    model = load_encoder(args.config, args.checkpoint, device)
    image_folder = ImageFolder(
        source, transform=build_transform(args.transform, args.resolution)
    )
    dataset = IndexedDataset(image_folder)
    sampler = DistributedSliceSampler(len(dataset), rank, world_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if rank == 0:
        print(
            f"Starting rank={rank}, world_size={world_size}, "
            f"dataset_size={len(dataset)}",
            flush=True,
        )
    return rank, world_size, device, model, loader


def get_latent_scaler(args):
    """Compute sampled-latent statistics without writing REPA dataset files."""
    rank, _, device, model, loader = _initialize_runtime(args)
    statistics = _compute_latent_statistics(
        model, loader, device, args.num_stat_samples
    )
    if rank == 0:
        print(
            "Latent statistics over "
            f"{statistics['num_samples']} samples: "
            f"mean={statistics['mean']}, std={statistics['std']}, "
            f"scaling_factor={statistics['scaling_factor']}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return statistics


def extract_repa_latents(args):
    """Encode the complete dataset and write files consumed by REPA."""
    if not args.dest:
        raise ValueError("--dest is required when --task=extract")

    rank, world_size, device, model, loader = _initialize_runtime(args)
    dest = Path(args.dest).expanduser()
    dest_images = (
        Path(args.dest_images).expanduser()
        if args.dest_images
        else dest.parent / "images"
    )
    if rank == 0:
        dest.mkdir(parents=True, exist_ok=True)
        if not args.no_images:
            dest_images.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    latent_labels = []
    image_labels = []
    for images, labels, indices in tqdm(
        loader, total=len(loader), disable=rank != 0
    ):
        images = images.to(device, non_blocking=True)
        with torch.inference_mode():
            posterior = model.encode(images)
            moments = torch.cat((posterior.mean, posterior.std), dim=1)

        images_cpu = (
            ((images.cpu().permute(0, 2, 3, 1).numpy() + 1.0) * 127.5)
            .clip(0, 255)
            .astype(np.uint8)
        )
        moments_cpu = moments.cpu().numpy().astype(np.float32, copy=False)
        entries = []
        for image, moments_item, label, index in zip(
            images_cpu, moments_cpu, labels.tolist(), indices.tolist()
        ):
            image_name, latent_name = _output_names(index)
            entries.append(
                (
                    latent_name,
                    moments_item,
                    image_name,
                    PIL.Image.fromarray(image),
                )
            )
            latent_labels.append([latent_name, label])
            if not args.no_images:
                image_labels.append([image_name, label])
        _save_batch(
            dest, dest_images, entries, save_images=not args.no_images
        )

    gathered_latent_labels = [None] * world_size if rank == 0 else None
    gathered_image_labels = [None] * world_size if rank == 0 else None
    dist.gather_object(latent_labels, gathered_latent_labels, dst=0)
    dist.gather_object(image_labels, gathered_image_labels, dst=0)
    if rank == 0:
        all_latent_labels = sorted(
            chain.from_iterable(gathered_latent_labels), key=lambda item: item[0]
        )
        with (dest / "dataset.json").open("w", encoding="utf-8") as handle:
            json.dump({"labels": all_latent_labels}, handle)
        if not args.no_images:
            all_image_labels = sorted(
                chain.from_iterable(gathered_image_labels),
                key=lambda item: item[0],
            )
            with (dest_images / "dataset.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump({"labels": all_image_labels}, handle)
        print(f"Extracted {len(all_latent_labels)} samples to {dest}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute VAE statistics or extract a REPA latent dataset."
    )
    parser.add_argument(
        "--task",
        choices=["extract", "stats"],
        default="extract",
        help="Extract REPA files or only compute sampled-latent statistics",
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest")
    parser.add_argument("--dest-images", default="")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--transform",
        choices=[
            "center-crop",
            "center-crop-wide",
            "center-crop-dhariwal",
        ],
        default="center-crop-dhariwal",
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--num-stat-samples",
        type=int,
        default=200000,
        help=(
            "Sample this many encoder latents to report their global mean, "
            "standard deviation, and reciprocal standard deviation"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-images", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.task == "stats":
        get_latent_scaler(arguments)
    else:
        extract_repa_latents(arguments)


# get statis
# torchrun --standalone --nproc_per_node=4 extract_REPA_latents.py \
#   --task=stats \
#   --source=/data/scratch/datasets/imagenet256/train \
#   --config=configs/sdvae_config_imgnet256.yaml \
#   --checkpoint=exps/imagenet256_SDVAE_ft/checkpoints/4000.ckpt \
#   --batch-size=128 \
#   --num-stat-samples=200000


# extract latents
# torchrun --standalone --nproc_per_node=4 extract_REPA_latents.py \
#   --task=extract \
#   --source=/data/scratch/datasets/imagenet256/train \
#   --dest=/data/scratch/datasets/imagenet256_repa/vae-sd \
#   --dest-images=/data/scratch/datasets/imagenet256_repa/images \
#   --config=configs/sdvae_config_imgnet256.yaml \
#   --checkpoint=exps/imagenet256_SDVAE_ft/checkpoints/4000.ckpt \
#   --batch-size=128