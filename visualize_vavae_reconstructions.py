"""Visualize ImageNet-256 inputs and VA-VAE reconstructions."""

import argparse
import os
from pathlib import Path

import numpy as np
import requests
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from core.config import instantiate_from_config


CHECKPOINT_URL = (
    "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/"
    "resolve/main/vavae-imagenet256-f16d32-dinov2.pt"
)


def center_crop_arr(image, image_size):
    """ADM center crop used by the released VA-VAE."""
    while min(*image.size) >= 2 * image_size:
        image = image.resize(
            tuple(size // 2 for size in image.size),
            resample=Image.Resampling.BOX,
        )
    scale = image_size / min(*image.size)
    image = image.resize(
        tuple(round(size * scale) for size in image.size),
        resample=Image.Resampling.BICUBIC,
    )
    array = np.asarray(image)
    crop_y = (array.shape[0] - image_size) // 2
    crop_x = (array.shape[1] - image_size) // 2
    return Image.fromarray(
        array[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def download_checkpoint(url, destination):
    destination = Path(destination).expanduser()
    if destination.is_file():
        print(f"Using cached checkpoint: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading VA-VAE checkpoint to {destination}")
    try:
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with partial.open("wb") as handle, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=destination.name,
            ) as progress:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        progress.update(len(chunk))
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def _checkpoint_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dictionary")
    state = checkpoint.get("model")
    if state is None:
        state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")
    if state and all(key.startswith("module.") for key in state):
        state = {
            key.removeprefix("module."): value
            for key, value in state.items()
        }
    return state


def load_vavae(config_path, checkpoint_path, device):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = _checkpoint_state(checkpoint)
    model_prefixes = (
        "encoder.",
        "decoder.",
        "quant_conv.",
        "post_quant_conv.",
    )
    state = {
        key: value
        for key, value in state.items()
        if key.startswith(model_prefixes)
    }
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        preview = ", ".join(incompatible.missing_keys[:10])
        raise RuntimeError(f"Checkpoint is missing VA-VAE weights: {preview}")
    print(
        f"Loaded VA-VAE ({sum(p.numel() for p in model.parameters()):,} "
        "parameters)"
    )
    return model.eval().requires_grad_(False).to(device)


def build_imagenet_loader(data_root, batch_size, num_workers):
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: center_crop_arr(image, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                inplace=True,
            ),
        ]
    )
    dataset = ImageFolder(Path(data_root).expanduser(), transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download VA-VAE, reconstruct ImageNet-256 images, and save a "
            "two-row input/reconstruction grid."
        )
    )
    parser.add_argument("--data-root", default="/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256/train", help="ImageFolder root (class subdirectories are required)")
    parser.add_argument("--config", type=Path, required=True, help="Path to the VA-VAE YAML configuration",)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path",)
    parser.add_argument("--output", type=Path, default=Path("vavae_reconstructions.png"))
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-posterior", action="store_true", help="Sample z instead of using the deterministic posterior mean")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = download_checkpoint(CHECKPOINT_URL, args.checkpoint)
    model = load_vavae(args.config, checkpoint, device)
    loader = build_imagenet_loader(args.data_root, args.num_images, args.num_workers)
    images, _ = next(iter(loader))
    images = images[: args.num_images].to(device, non_blocking=True)
    reconstructions = model.reconstruct(images, sample_posterior=args.sample_posterior)

    # Batch order plus nrow=N gives inputs on row 1 and reconstructions on row 2.
    comparison = torch.cat([images, reconstructions], dim=0)
    comparison = comparison.clamp(-1, 1).add(1).div(2).cpu()
    grid = make_grid(comparison, nrow=images.shape[0], padding=2)
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output)
    print(
        f"Saved {images.shape[0]} inputs (row 1) and reconstructions "
        f"(row 2) to {output.resolve()}"
    )


if __name__ == "__main__":
    main()


# python visualize_vavae_reconstructions.py \
#   --config /home/ningmang/eqvae/configs/VAVAE_config_imgnet256.yaml \
#   --checkpoint /home/ningmang/eqvae/exps/imagenet256_VAVAE/vavae-imagenet256-f16d32-dinov2.pt \
#   --output vavae_reconstructions.png