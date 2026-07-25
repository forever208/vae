"""Distributed reconstruction metrics and validation."""

import torch
from torchmetrics.functional.image.ssim import (
    structural_similarity_index_measure,
)
from torchmetrics.image.fid import FrechetInceptionDistance

from .builders import move_batch_to_device
from .logging import log_reconstruction_images


class ReconstructionMetrics:
    def __init__(self, device):
        self.fid = FrechetInceptionDistance(
            feature=2048,
            reset_real_features=False,
            normalize=False,
            sync_on_compute=True,
        ).to(device)
        self.real_features_cached = False
        self.device = device

    def begin(self):
        self.sums = torch.zeros(3, dtype=torch.float64, device=self.device)
        self.samples = torch.zeros((), dtype=torch.long, device=self.device)

    @torch.no_grad()
    def update(self, inputs, reconstructions, perceptual_loss):
        real = torch.clamp(127.5 * inputs + 128.0, 0, 255).to(torch.uint8)
        fake = torch.clamp(127.5 * reconstructions + 128.0, 0, 255).to(
            torch.uint8
        )
        if not self.real_features_cached:
            self.fid.update(real, real=True)
        self.fid.update(fake, real=False)
        batch_size = inputs.shape[0]
        mse = (real.float() - fake.float()).pow(2).flatten(1).mean(1)
        psnr = 20.0 * torch.log10(
            255.0 / torch.sqrt(mse.clamp_min(1e-12))
        )
        lpips = perceptual_loss(
            inputs.contiguous(), reconstructions.contiguous()
        ).reshape(batch_size, -1).mean(1)
        ssim = structural_similarity_index_measure(
            reconstructions.float(),
            inputs.float(),
            data_range=2.0,
            reduction="none",
        )
        self.sums += torch.stack(
            [psnr.sum(), lpips.double().sum(), ssim.double().sum()]
        )
        self.samples += batch_size

    def finalize(self, distributed_context):
        fid = float(self.fid.compute().detach().cpu())
        sums = distributed_context.all_reduce_sum(self.sums)
        samples = distributed_context.all_reduce_sum(self.samples)
        averages = sums / samples.clamp_min(1)
        metrics = {
            "val/fid": fid,
            "val/psnr": float(averages[0]),
            "val/lpips": float(averages[1]),
            "val/ssim": float(averages[2]),
            "samples": int(samples),
        }
        self.real_features_cached = True
        self.fid.reset()
        return metrics


@torch.no_grad()
def run_validation(
    model,
    val_loader,
    metrics,
    device,
    distributed_context,
    config,
    local_step,
    global_step,
):
    distributed_context.barrier()
    if distributed_context.is_main:
        print(
            f"Starting eval: local_step={local_step} "
            f"global_step={global_step}",
            flush=True,
        )
    was_training = model.training
    model.eval()
    metrics.begin()
    for batch_index, batch in enumerate(val_loader):
        batch = move_batch_to_device(batch, device)
        outputs = model.compute_validation_outputs(batch)
        metrics.update(
            outputs["inputs"],
            outputs["reconstructions"],
            model.loss.perceptual_loss,
        )
        if (
            batch_index == 0
            and distributed_context.is_main
            and config.pytorch_training.logging.validation_images
        ):
            log_reconstruction_images(
                outputs["inputs"],
                outputs["reconstructions"],
                local_step,
                config,
            )
    results = metrics.finalize(distributed_context)
    if was_training:
        model.train()
    distributed_context.barrier()
    return results
