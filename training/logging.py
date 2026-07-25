"""Scalar, evaluation, progress, and reconstruction image logging."""

import datetime
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image


def to_scalars(values):
    result = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            result[key] = float(value.detach().cpu())
        elif isinstance(value, (float, int)):
            result[key] = float(value)
    return result


def parameter_norms(model, names):
    parameters = dict(model.named_parameters())
    return {
        f"parameter_norm/{name}": float(
            parameters[name].detach().float().norm().cpu()
        )
        for name in names
    }


def build_train_metrics(
    model, training, local_step, global_step, ae_result, disc_result
):
    ae_loss, ae_logs, ae_meta = ae_result
    disc_loss, disc_logs, disc_meta = disc_result
    metrics = {
        "local_step": local_step,
        "global_step": global_step,
        "train/ae_loss": float(ae_loss.detach().cpu()),
        "train/disc_loss": float(disc_loss.detach().cpu()),
        "train/ae_scale": float(ae_meta["scale"].cpu()),
        "train/ae_latent": float(ae_meta["latent"].cpu()),
        "train/disc_scale": float(disc_meta["scale"].cpu()),
        "train/disc_latent": float(disc_meta["latent"].cpu()),
    }
    metrics.update(to_scalars(ae_logs))
    metrics.update(to_scalars(disc_logs))
    if training.logging.print_parameter_norms:
        metrics.update(
            parameter_norms(model, training.comparison.parameter_names)
        )
    return metrics


def write_json_line(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(values, sort_keys=True) + "\n")


def progress_values(metrics, optimizers):
    return {
        "lstep": metrics["local_step"],
        "gstep": metrics["global_step"],
        "lr": f"{optimizers[0].param_groups[0]['lr']:.2g}",
        "ae": f"{metrics['train/ae_loss']:.3g}",
        "rec": f"{metrics['train/weighted_pixel_loss']:.3g}",
        "lpips": f"{metrics['train/weighted_lpips_loss']:.3g}",
        "gan": f"{metrics['train/weighted_g_loss']:.3g}",
        "kl": f"{metrics['train/weighted_kl_loss']:.3g}",
    }


def write_evaluation(path, metrics, local_step, global_step):
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    line = (
        f"{timestamp} local_step={local_step} global_step={global_step} "
        f"samples={metrics['samples']} FID={metrics['val/fid']:.6f} "
        f"PSNR={metrics['val/psnr']:.6f} "
        f"LPIPS={metrics['val/lpips']:.6f} "
        f"SSIM={metrics['val/ssim']:.6f}\n"
    )
    with path.open("a", encoding="utf-8") as output:
        output.write(line)


def save_image_grid(grid, path):
    array = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        (array * 255).clip(0, 255).astype(np.uint8)
    ).save(path)


def log_reconstruction_images(
    inputs, reconstructions, local_step, config
):
    inputs = inputs[:4].detach().cpu().clamp(-1, 1)
    reconstructions = reconstructions[:4].detach().cpu().clamp(-1, 1)
    images = torch.cat((inputs, reconstructions), dim=0)
    grid = torchvision.utils.make_grid(images, nrow=4)
    normalized = (grid + 1.0) / 2.0
    output_dir = Path(config.pytorch_training.checkpoint.output_dir)
    save_image_grid(normalized, output_dir / f"{local_step}.png")
