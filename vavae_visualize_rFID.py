"""Visualize VA-VAE reconstructions and evaluate their ImageNet-256 rFID."""

import argparse
import os
from pathlib import Path

import requests
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from core.config import instantiate_from_config
from data.datasets import DatasetVal


CHECKPOINT_URL = (
    "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/"
    "resolve/main/vavae-imagenet256-f16d32-dinov2.pt"
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
        state = {key.removeprefix("module."): value for key, value in state.items()}
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
    state = {key: value for key, value in state.items() if key.startswith(model_prefixes)}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        preview = ", ".join(incompatible.missing_keys[:10])
        raise RuntimeError(f"Checkpoint is missing VA-VAE weights: {preview}")
    print(f"Loaded VA-VAE ({sum(p.numel() for p in model.parameters()):,} parameters)")
    return model.eval().requires_grad_(False).to(device)


def build_imagenet_loader(data_root, batch_size, num_workers):
    dataset = DatasetVal(
        val_dir=str(Path(data_root).expanduser()),
        dataset_name="imagenet",
        size=256,
        degradation="pil_nearest",
        eval_preprocessing=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.inference_mode()
def calculate_rfid_and_examples(model, loader, device, num_images, sample_posterior):
    """Compute rFID exactly like training validation and retain examples for the grid."""
    fid = FrechetInceptionDistance(
        feature=2048,
        reset_real_features=False,
        normalize=False,
        sync_on_compute=True,
    ).to(device)
    example_inputs = []
    example_reconstructions = []
    collected = 0

    for batch in tqdm(loader, desc="Computing rFID", dynamic_ncols=True):
        images = batch["image"].permute(0, 3, 1, 2).contiguous()
        images = images.to(device, non_blocking=True)
        reconstructions = model.reconstruct(images, sample_posterior=sample_posterior)
        real = torch.clamp(127.5 * images + 128.0, 0, 255).to(torch.uint8)
        fake = torch.clamp(127.5 * reconstructions + 128.0, 0, 255).to(torch.uint8)
        fid.update(real, real=True)
        fid.update(fake, real=False)

        if collected < num_images:
            keep = min(num_images - collected, images.shape[0])
            example_inputs.append(images[:keep].cpu())
            example_reconstructions.append(reconstructions[:keep].cpu())
            collected += keep

    if collected < num_images:
        raise ValueError(f"--num-images={num_images} exceeds dataset size {collected}")
    examples = torch.cat(example_inputs, dim=0)
    reconstructions = torch.cat(example_reconstructions, dim=0)
    return float(fid.compute().cpu()), examples, reconstructions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate VA-VAE rFID and save a two-row input/reconstruction grid."
    )
    parser.add_argument(
        "--data-root",
        default="/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256/val",
        help="ImageFolder root containing the evaluation images",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the VA-VAE YAML configuration",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint path",
    )
    parser.add_argument("--output", type=Path, default=Path("vavae_reconstructions.png"))
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    posterior_group = parser.add_mutually_exclusive_group()
    posterior_group.add_argument(
        "--sample-posterior",
        dest="sample_posterior",
        action="store_true",
        help="Sample posterior latents, matching SDVAE/EQVAE training evaluation",
    )
    posterior_group.add_argument(
        "--no-sample-posterior",
        dest="sample_posterior",
        action="store_false",
        help="Use deterministic posterior means instead of training evaluation semantics",
    )
    parser.set_defaults(sample_posterior=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = download_checkpoint(CHECKPOINT_URL, args.checkpoint)
    model = load_vavae(args.config, checkpoint, device)
    loader = build_imagenet_loader(args.data_root, args.eval_batch_size, args.num_workers)
    rfid, images, reconstructions = calculate_rfid_and_examples(
        model,
        loader,
        device,
        args.num_images,
        args.sample_posterior,
    )
    print(f"rFID: {rfid:.6f}")

    # Batch order plus nrow=N gives inputs on row 1 and reconstructions on row 2.
    comparison = torch.cat([images, reconstructions], dim=0)
    comparison = comparison.clamp(-1, 1).add(1).div(2)
    grid = make_grid(comparison, nrow=images.shape[0], padding=2)
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output)
    print(f"Saved {images.shape[0]} inputs (row 1) and reconstructions (row 2) to "
          f"{output.resolve()}")


if __name__ == "__main__":
    main()


# python visualize_vavae_reconstructions.py \
#   --config /home/ningmang/eqvae/configs/VAVAE_config_imgnet256.yaml \
#   --checkpoint /home/ningmang/eqvae/exps/imagenet256_VAVAE/vavae-imagenet256-f16d32-dinov2.pt \
#   --data-root /mnt/vepfs/base2/stream/ningmang/datasets/imagenet256/val \
#   --output vavae_reconstructions.png