"""Visualize VAE reconstructions and evaluate rFID, PSNR, and lpFID."""

import argparse
import os
from pathlib import Path

import requests
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Sampler, Subset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from core.config import instantiate_from_config
from data.datasets import DatasetVal
from training.distributed import DistributedContext, configure_reproducibility


CHECKPOINT_URL = (
    "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/"
    "resolve/main/vavae-imagenet256-f16d32-dinov2.pt"
)
LOW_PASS_BLOCK_SIZE = 8


class DistributedEvalSampler(Sampler):
    """Shard a dataset across ranks without padding or duplicate samples."""

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.world_size - 1) // self.world_size)


def resolve_checkpoint(destination, url=None):
    destination = Path(destination).expanduser()
    if destination.is_file():
        print(f"Using checkpoint: {destination}")
        return destination

    if url is None:
        raise FileNotFoundError(
            f"Checkpoint does not exist: {destination}. "
            "Pass --checkpoint-url to download it explicitly."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading checkpoint to {destination}")
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
    state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_vae(config_path, checkpoint_path, device, verbose=True):
    config = OmegaConf.load(config_path)
    model_config = OmegaConf.create(
        OmegaConf.to_container(config.model, resolve=True)
    )
    if "params" not in model_config:
        model_config.params = {}
    # This in-memory override avoids constructing training-only losses. It
    # does not modify the YAML or any model/training implementation.
    model_config.params.inference_only = True
    model = instantiate_from_config(model_config)
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
    model.load_state_dict(state, strict=True)
    if verbose:
        print(
            f"Loaded {model_config.target} "
            f"({sum(p.numel() for p in model.parameters()):,} parameters)"
        )
    return model.eval().requires_grad_(False).to(device)


def build_imagenet_loader(
    data_root, batch_size, num_workers, distributed_context=None
):
    dataset = DatasetVal(
        val_dir=str(Path(data_root).expanduser()),
        dataset_name="imagenet",
        size=256,
        degradation="pil_nearest",
        eval_preprocessing=True,
    )
    sampler = (
        DistributedEvalSampler(
            dataset,
            distributed_context.rank,
            distributed_context.world_size,
        )
        if distributed_context is not None
        and distributed_context.distributed
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.inference_mode()
def reconstruct(model, images, sample_posterior):
    """Reconstruct through the encode/decode API shared by supported VAEs."""
    posterior = model.encode(images)
    latents = posterior.sample() if sample_posterior else posterior.mode()
    return model.decode(latents)


def calculate_batch_psnr(real, fake):
    """Return per-image PSNR for uint8 images, matching training validation."""
    mse = (real.float() - fake.float()).pow(2).flatten(1).mean(1)
    return 20.0 * torch.log10(
        255.0 / torch.sqrt(mse.clamp_min(1e-12))
    )


def normalize_lpfid_masks(masks):
    """Validate and deduplicate low-pass masks while preserving their order."""
    normalized = []
    seen = set()
    for mask in masks:
        if not 0 <= mask <= 14:
            raise ValueError("--lpfid-masks values must be between 0 and 14")
        if mask not in seen:
            normalized.append(mask)
            seen.add(mask)
    return normalized


def low_pass(tensor, mask, block_size=LOW_PASS_BLOCK_SIZE):
    """Apply the same block-DCT low-pass filter used by DSM regularization."""
    if mask == 0:
        return tensor

    from utils_DCT import (
        combine_blocks_torch,
        dct_2d_torch_unified,
        idct_2d_torch_unified,
        split_into_blocks_torch,
    )

    height, width = tensor.shape[-2:]
    if height % block_size != 0 or width % block_size != 0:
        raise ValueError(
            f"Block size {block_size} must divide tensor shape "
            f"{height}x{width}"
        )

    blocks = split_into_blocks_torch(tensor, block_size)
    coefficients = dct_2d_torch_unified(blocks, center="none")
    max_sum = 2 * (block_size - 1)
    threshold = max_sum - (mask - 1)
    u = torch.arange(block_size, device=tensor.device).view(block_size, 1)
    v = torch.arange(block_size, device=tensor.device).view(1, block_size)
    coefficients[..., (u + v) >= threshold] = 0.0
    blocks = idct_2d_torch_unified(coefficients, center="none")
    return combine_blocks_torch(blocks, height, width, block_size)


@torch.inference_mode()
def calculate_rfid_psnr(
    model, loader, device, sample_posterior, distributed_context
):
    """Compute globally synchronized validation rFID and PSNR."""
    fid = FrechetInceptionDistance(
        feature=2048,
        reset_real_features=False,
        normalize=False,
        sync_on_compute=True,
    ).to(device)
    psnr_sum = torch.zeros((), dtype=torch.float64, device=device)
    sample_count = torch.zeros((), dtype=torch.long, device=device)

    for batch in tqdm(
        loader,
        desc="Computing rFID/PSNR",
        dynamic_ncols=True,
        disable=not distributed_context.is_main,
    ):
        images = batch["image"].permute(0, 3, 1, 2).contiguous()
        images = images.to(device, non_blocking=True)
        reconstructions = reconstruct(model, images, sample_posterior)
        real = torch.clamp(127.5 * images + 128.0, 0, 255).to(torch.uint8)
        fake = torch.clamp(127.5 * reconstructions + 128.0, 0, 255).to(torch.uint8)
        fid.update(real, real=True)
        fid.update(fake, real=False)
        psnr = calculate_batch_psnr(real, fake)
        psnr_sum += psnr.double().sum()
        sample_count += images.shape[0]

    rfid = float(fid.compute().detach().cpu())
    psnr_sum = distributed_context.all_reduce_sum(psnr_sum)
    sample_count = distributed_context.all_reduce_sum(sample_count)
    mean_psnr = float((psnr_sum / sample_count.clamp_min(1)).cpu())
    return rfid, mean_psnr


@torch.inference_mode()
def collect_examples(
    model,
    dataset,
    device,
    num_images,
    batch_size,
    sample_posterior,
):
    """Reconstruct the first dataset samples on rank zero for visualization."""
    loader = DataLoader(
        Subset(dataset, range(num_images)),
        batch_size=min(batch_size, num_images),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    example_inputs = []
    example_reconstructions = []
    for batch in loader:
        images = batch["image"].permute(0, 3, 1, 2).contiguous()
        images = images.to(device, non_blocking=True)
        reconstructions = reconstruct(model, images, sample_posterior)
        example_inputs.append(images.cpu())
        example_reconstructions.append(reconstructions.cpu())
    return (
        torch.cat(example_inputs, dim=0),
        torch.cat(example_reconstructions, dim=0),
    )


@torch.inference_mode()
def calculate_lpfid(
    model,
    loader,
    device,
    mask,
    sample_posterior,
    distributed_context,
):
    """Compute FID between low-pass inputs and low-pass-latent reconstructions."""
    fid = FrechetInceptionDistance(
        feature=2048,
        reset_real_features=False,
        normalize=False,
        sync_on_compute=True,
    ).to(device)

    for batch in tqdm(
        loader,
        desc=f"Computing lpFID (mask={mask})",
        dynamic_ncols=True,
        disable=not distributed_context.is_main,
    ):
        images = batch["image"].permute(0, 3, 1, 2).contiguous()
        images = images.to(device, non_blocking=True)
        posterior = model.encode(images)
        latents = (
            posterior.sample() if sample_posterior else posterior.mode()
        )
        low_pass_images = low_pass(images, mask)
        low_pass_latents = low_pass(latents, mask)
        reconstructions = model.decode(low_pass_latents)
        real = torch.clamp(
            127.5 * low_pass_images + 128.0, 0, 255
        ).to(torch.uint8)
        fake = torch.clamp(
            127.5 * reconstructions + 128.0, 0, 255
        ).to(torch.uint8)
        fid.update(real, real=True)
        fid.update(fake, real=False)

    return float(fid.compute().detach().cpu())


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VAE rFID, PSNR, and optional lpFID, then save a "
            "two-row input/reconstruction grid."
        )
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
        help="Path to the VAE YAML configuration",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint path",
    )
    parser.add_argument(
        "--checkpoint-url",
        help=(
            "Optional URL used only when --checkpoint does not exist. "
            f"Released VA-VAE URL: {CHECKPOINT_URL}"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("vavae_reconstructions.png"))
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Evaluation batch size per process/GPU",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers per process",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--distributed-backend",
        default="nccl" if torch.cuda.is_available() else "gloo",
        choices=("nccl", "gloo"),
        help="torch.distributed backend used when launched with torchrun",
    )
    parser.add_argument(
        "--lpfid-masks",
        type=int,
        nargs="*",
        default=[],
        metavar="MASK",
        help=(
            "DSM low-pass masks to evaluate with block size 8; "
            "omit or leave empty to disable lpFID"
        ),
    )
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
    lpfid_masks = normalize_lpfid_masks(args.lpfid_masks)
    distributed_context = DistributedContext(args.distributed_backend)
    try:
        configure_reproducibility(
            args.seed,
            deterministic=False,
            benchmark=True,
            rank=distributed_context.rank,
        )
        device = (
            torch.device("cuda", distributed_context.local_rank)
            if distributed_context.distributed and torch.cuda.is_available()
            else torch.device(args.device)
        )
        if device.type == "cuda":
            torch.cuda.set_device(device)

        checkpoint = Path(args.checkpoint).expanduser()
        if distributed_context.is_main:
            checkpoint = resolve_checkpoint(
                checkpoint,
                args.checkpoint_url,
            )
        distributed_context.barrier()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

        model = load_vae(
            args.config,
            checkpoint,
            device,
            verbose=distributed_context.is_main,
        )
        loader = build_imagenet_loader(
            args.data_root,
            args.eval_batch_size,
            args.num_workers,
            distributed_context,
        )
        if args.num_images > len(loader.dataset):
            raise ValueError(
                f"--num-images={args.num_images} exceeds dataset size "
                f"{len(loader.dataset)}"
            )

        rfid, psnr = calculate_rfid_psnr(
            model,
            loader,
            device,
            args.sample_posterior,
            distributed_context,
        )
        lpfid_results = {}
        for mask in lpfid_masks:
            lpfid_results[mask] = (
                rfid
                if mask == 0
                else calculate_lpfid(
                    model,
                    loader,
                    device,
                    mask,
                    args.sample_posterior,
                    distributed_context,
                )
            )

        if distributed_context.is_main:
            print(f"rFID: {rfid:.6f}")
            print(f"PSNR: {psnr:.6f} dB")
            for mask, lpfid in lpfid_results.items():
                print(f"lpFID(mask={mask}): {lpfid:.6f}")

            torch.manual_seed(args.seed)
            images, reconstructions = collect_examples(
                model,
                loader.dataset,
                device,
                args.num_images,
                args.eval_batch_size,
                args.sample_posterior,
            )
            comparison = torch.cat([images, reconstructions], dim=0)
            comparison = comparison.clamp(-1, 1).add(1).div(2)
            grid = make_grid(
                comparison,
                nrow=images.shape[0],
                padding=2,
            )
            output = args.output.expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            save_image(grid, output)
            print(
                f"Saved {images.shape[0]} inputs (row 1) and "
                f"reconstructions (row 2) to {output.resolve()}"
            )
        distributed_context.barrier()
    finally:
        distributed_context.close()


if __name__ == "__main__":
    main()