"""Extract CelebA256 VAE posterior moments for DCTdiff training."""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from tqdm import tqdm

from core.config import get_obj_from_str


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs/SDVAE_config_celeba256.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class CelebA256WithFlip(Dataset):
    """Return every source image once normally and once horizontally flipped."""

    def __init__(self, source, resolution):
        self.source = Path(source).expanduser()
        if not self.source.is_dir():
            raise FileNotFoundError(f"Source dataset does not exist: {self.source}")

        self.files = sorted(
            path
            for path in self.source.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.files:
            raise RuntimeError(f"No images found in dataset directory: {self.source}")

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (resolution, resolution),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )

    @property
    def source_size(self):
        return len(self.files)

    def __len__(self):
        return 2 * self.source_size

    def __getitem__(self, index):
        source_index = index % self.source_size
        flipped = index >= self.source_size
        with PIL.Image.open(self.files[source_index]) as image:
            image = image.convert("RGB")
            if flipped:
                image = image.transpose(PIL.Image.Transpose.FLIP_LEFT_RIGHT)
            return self.transform(image), index


class DistributedSliceSampler(Sampler):
    """Shard indices across ranks without adding duplicate padding samples."""

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
    model_params = OmegaConf.to_container(
        config.model.get("params", {}), resolve=True
    ) or {}
    model_params["encoder_only"] = True
    if "inference_only" in model_params:
        model_params["inference_only"] = False
    model_class = get_obj_from_str(config.model.target)
    model = model_class(**model_params)

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if state is None and checkpoint and all(
        torch.is_tensor(value) for value in checkpoint.values()
    ):
        state = checkpoint
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

    return model.eval().requires_grad_(False).to(device)


def _initialize_distributed(seed):
    if not torch.cuda.is_available():
        raise RuntimeError("Latent extraction requires at least one CUDA GPU")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Distributed environment is not initialized; launch with torchrun"
        )

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(seed * world_size + rank)
    return rank, world_size, device


def _save_batch(destination, indices, moments, executor):
    futures = [
        executor.submit(np.save, destination / f"{index}.npy", moment)
        for index, moment in zip(indices, moments)
    ]
    for future in futures:
        future.result()


@torch.inference_mode()
def extract_latents(args):
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.num_stat_samples <= 0:
        raise ValueError("--num-stat-samples must be greater than zero")

    rank, world_size, device = _initialize_distributed(args.seed)
    try:
        dataset = CelebA256WithFlip(args.source, args.resolution)
        if args.expected_source_size and dataset.source_size != args.expected_source_size:
            raise RuntimeError(
                f"Expected {args.expected_source_size} source images, "
                f"but found {dataset.source_size}"
            )

        sampler = DistributedSliceSampler(len(dataset), rank, world_size)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            drop_last=False,
        )
        model = load_encoder(args.config, args.checkpoint, device)
        destination = Path(args.dest).expanduser()
        if rank == 0:
            destination.mkdir(parents=True, exist_ok=True)
            print(
                f"Found {dataset.source_size} CelebA256 images; "
                f"extracting {len(dataset)} original/flip posterior files "
                f"with world_size={world_size}",
                flush=True,
            )
        dist.barrier()

        total_stat_samples = min(args.num_stat_samples, len(dataset))
        value_sum = torch.zeros((), dtype=torch.float64, device=device)
        square_sum = torch.zeros((), dtype=torch.float64, device=device)
        value_count = torch.zeros((), dtype=torch.int64, device=device)
        stat_sample_count = torch.zeros((), dtype=torch.int64, device=device)
        saved_count = torch.zeros((), dtype=torch.int64, device=device)
        save_workers = max(1, min(args.save_workers, os.cpu_count() or 1))

        with ThreadPoolExecutor(max_workers=save_workers) as executor:
            for images, indices in tqdm(
                loader,
                total=len(loader),
                desc="Extracting latents",
                disable=rank != 0,
            ):
                images = images.to(device, non_blocking=True)
                posterior = model.encode(images)

                moments = (
                    posterior.parameters.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                index_list = indices.tolist()
                _save_batch(destination, index_list, moments, executor)
                saved_count += len(index_list)

                stat_mask = indices < total_stat_samples
                if stat_mask.any():
                    sampled = posterior.sample()[stat_mask.to(device)]
                    sampled = sampled.to(torch.float64).reshape(-1)
                    value_sum += sampled.sum()
                    square_sum += sampled.square().sum()
                    value_count += sampled.numel()
                    stat_sample_count += stat_mask.sum().to(device)

        for statistic in (
            value_sum,
            square_sum,
            value_count,
            stat_sample_count,
            saved_count,
        ):
            dist.all_reduce(statistic, op=dist.ReduceOp.SUM)

        if rank == 0:
            if saved_count.item() != len(dataset):
                raise RuntimeError(
                    f"Saved {saved_count.item()} files, expected {len(dataset)}"
                )
            mean = value_sum / value_count
            variance = torch.clamp(
                square_sum / value_count - mean.square(), min=0.0
            )
            std = torch.sqrt(variance)
            std_value = std.item()
            scaler = 1.0 / std_value if std_value > 0 else float("nan")
            print(
                f"Latent statistics over {stat_sample_count.item()} samples: "
                f"mean={mean.item()}, std={std_value}, scaler={scaler}",
                flush=True,
            )
            print(f"Saved {saved_count.item()} files to {destination}", flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract CelebA256 mean/logvar arrays consumed by DCTdiff feature "
            "datasets."
        )
    )
    parser.add_argument("--source", required=True, help="Flat CelebA256 image directory")
    parser.add_argument("--dest", required=True, help="Output directory for numbered .npy files")
    parser.add_argument("--checkpoint", required=True, help="VAE .ckpt checkpoint")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-workers", type=int, default=8)
    parser.add_argument("--num-stat-samples", type=int, default=50000)
    parser.add_argument("--expected-source-size", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    extract_latents(parse_args())


# Example:
# torchrun --standalone --nproc_per_node=8 extract_latents.py \
#   --source=/mnt/vepfs/base2/stream/ningmang/datasets/celeba256/celeba256 \
#   --dest=/mnt/vepfs/base2/stream/ningmang/datasets/celeba256_latents/my_vae \
#   --config=configs/SDVAE_config_celeba256.yaml \
#   --checkpoint=exps/celeba256_SDVAE/checkpoints/100000.ckpt \
#   --batch-size=128 \
#   --num-stat-samples=50000
