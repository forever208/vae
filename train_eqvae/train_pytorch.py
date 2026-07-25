"""EQ-VAE training: single GPU/DDP, validation, TensorBoard, images and top-k checkpoints."""

import argparse
import contextlib
import datetime
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from ldm.util import instantiate_from_config


# --- configuration and distributed lifecycle -----------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch trainer for EQ-VAE")
    parser.add_argument("--config", default="configs/eqvae_config.yaml", help="Complete YAML training configuration")
    return parser.parse_args()


class DistributedContext:
    def __init__(self, backend):
        self.rank, self.local_rank, self.world_size = (int(os.environ.get(name, default)) for name, default in
                                                       (("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1)))
        self.distributed = self.world_size > 1
        if self.distributed:
            if backend == "nccl":
                if not torch.cuda.is_available():
                    raise RuntimeError("NCCL distributed training requires CUDA")
                torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend=backend, init_method="env://")

    @property
    def is_main(self):
        return self.rank == 0

    def barrier(self):
        if self.distributed:
            dist.barrier()

    def all_reduce_sum(self, value):
        if self.distributed:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def close(self):
        if self.distributed:
            dist.barrier()
            dist.destroy_process_group()


def configure_reproducibility(seed, deterministic, benchmark, rank):
    process_seed = int(seed) + int(rank)
    random.seed(process_seed); np.random.seed(process_seed); torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = bool(deterministic), bool(benchmark)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(training, distributed_context):
    use_local_cuda = distributed_context.distributed and torch.cuda.is_available()
    device = torch.device("cuda", distributed_context.local_rank) if use_local_cuda else torch.device(training.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    return device


def calculate_learning_rate(config, world_size):
    base_lr, training = float(config.model.base_learning_rate), config.pytorch_training
    return base_lr if not training.scale_learning_rate else base_lr * int(config.data.batch_size) * world_size * int(training.accumulate_grad_batches)


# --- data, model and optimizer construction -------------------------------------------------------

def build_loader(dataset_config, config, distributed_context, dataloader_config, shuffle):
    dataset = instantiate_from_config(dataset_config)
    sampler = DistributedSampler(dataset, distributed_context.world_size, distributed_context.rank, shuffle=shuffle,
                                 seed=int(config.pytorch_training.seed), drop_last=bool(dataloader_config.drop_last)) \
        if distributed_context.distributed else None
    loader = DataLoader(dataset, batch_size=int(config.data.batch_size), sampler=sampler, shuffle=shuffle if sampler is None else False,
                        num_workers=int(dataloader_config.num_workers), pin_memory=bool(dataloader_config.pin_memory),
                        drop_last=bool(dataloader_config.drop_last) if sampler is None else False)
    return loader, sampler


def build_data_loaders(config, distributed_context):
    training = config.pytorch_training
    train_loader, train_sampler = build_loader(config.data.train, config, distributed_context, training.dataloader,
                                               bool(training.dataloader.shuffle))
    val_loader, _ = build_loader(config.data.validation, config, distributed_context, training.validation.dataloader, False)
    return train_loader, train_sampler, val_loader


class DistributedTrainStep(torch.nn.Module):
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, batch, optimizer_index, global_step):
        return self.autoencoder.compute_train_loss(batch, optimizer_index, global_step)


def build_model_and_optimizers(config, device, distributed_context):
    training = config.pytorch_training
    autoencoder = instantiate_from_config(config.model).to(device)
    learning_rate = calculate_learning_rate(config, distributed_context.world_size)
    optimizers = autoencoder.build_optimizers(learning_rate, betas=tuple(float(beta) for beta in training.optimizer.betas))
    train_model = DistributedTrainStep(autoencoder).to(device)
    if distributed_context.distributed:
        train_model = DDP(train_model, device_ids=[distributed_context.local_rank], output_device=distributed_context.local_rank,
                          find_unused_parameters=bool(training.distributed.find_unused_parameters),
                          broadcast_buffers=bool(training.distributed.broadcast_buffers))
    return autoencoder, train_model, optimizers, learning_rate


def move_batch_to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


# --- checkpoint I/O and top-k --------------------------------------------------------------------

def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(path, model, optimizers, device, strict_model_load, resume_optimizer):
    checkpoint = torch_load(path, device)
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError(f"Checkpoint {path} has no model or state_dict entry")
    if state and next(iter(state)).startswith("module."):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=bool(strict_model_load))
    if not strict_model_load and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(f"Checkpoint incompatibilities: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}")
    if resume_optimizer:
        states = checkpoint.get("optimizer_states", [])
        if len(states) != len(optimizers):
            raise ValueError(f"Expected {len(optimizers)} optimizer states, found {len(states)} in {path}")
        for optimizer, optimizer_state in zip(optimizers, states):
            optimizer.load_state_dict(optimizer_state)
    return {"global_step": int(checkpoint.get("global_step", 0)), "epoch": int(checkpoint.get("epoch", 0)),
            "topk": checkpoint.get("topk", [])}


def capture_rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint(path, model, optimizers, global_step, epoch, local_step, config, topk):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 2, "model": model.state_dict(), "optimizer_states": [optimizer.state_dict() for optimizer in optimizers],
                "global_step": global_step, "epoch": epoch, "local_step": local_step, "topk": topk, "rng_state": capture_rng_state(),
                "config": OmegaConf.to_container(config, resolve=True)}, path)


class CheckpointManager:
    def __init__(self, config, resume_topk=None):
        cfg = config.pytorch_training.checkpoint
        self.config, self.directory = config, Path(cfg.output_dir) / cfg.directory
        self.monitor, self.mode, self.save_top_k, self.save_last = str(cfg.monitor), str(cfg.mode), int(cfg.save_top_k), bool(cfg.save_last)
        self.filename_pattern, self.entries = str(cfg.filename_pattern), list(resume_topk or [])

    def save_last_checkpoint(self, model, optimizers, global_step, epoch, local_step):
        if self.save_last:
            save_checkpoint(self.directory / "last.pt", model, optimizers, global_step, epoch, local_step, self.config, self.entries)

    def update(self, metrics, model, optimizers, global_step, epoch, local_step):
        score = float(metrics[self.monitor])
        candidate = {"score": score, "global_step": global_step,
                     "path": str(self.directory / self.filename_pattern.format(global_step=global_step, epoch=epoch, score=score))}
        should_keep = self.save_top_k < 0 or len(self.entries) < self.save_top_k
        if not should_keep and self.entries:
            worst = (max if self.mode == "min" else min)(self.entries, key=lambda item: item["score"])
            should_keep = score < worst["score"] if self.mode == "min" else score > worst["score"]
            if should_keep:
                Path(worst["path"]).unlink(missing_ok=True); self.entries.remove(worst)
        if should_keep:
            self.entries.append(candidate); self.entries.sort(key=lambda item: item["score"], reverse=self.mode == "max")
            save_checkpoint(candidate["path"], model, optimizers, global_step, epoch, local_step, self.config, self.entries)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "topk.json").write_text(json.dumps(self.entries, indent=2, sort_keys=True), encoding="utf-8")
        self.save_last_checkpoint(model, optimizers, global_step, epoch, local_step)


# --- optimizer mechanics, scalar and image logging ------------------------------------------------

@contextlib.contextmanager
def activate_optimizer(optimizers, active_index):
    requires_grad = {}
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter not in requires_grad:
                    requires_grad[parameter], parameter.requires_grad = parameter.requires_grad, False
    for group in optimizers[active_index].param_groups:
        for parameter in group["params"]:
            parameter.requires_grad = requires_grad[parameter]
    try:
        yield
    finally:
        for parameter, original in requires_grad.items():
            parameter.requires_grad = original


def optimizer_pass(train_model, optimizers, batch, optimizer_index, global_step, local_step, accumulation, should_step,
                   zero_grad_set_to_none, distributed_context):
    optimizer = optimizers[optimizer_index]
    if local_step % accumulation == 0:
        optimizer.zero_grad(set_to_none=bool(zero_grad_set_to_none))
    sync_context = contextlib.nullcontext() if should_step or not distributed_context.distributed else train_model.no_sync()
    with sync_context, activate_optimizer(optimizers, optimizer_index):
        loss, log_dict, metadata = train_model(batch, optimizer_index, global_step)
        (loss / accumulation).backward()
    if should_step:
        optimizer.step(); global_step += 1
    return loss, log_dict, metadata, global_step


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
    return {f"parameter_norm/{name}": float(parameters[name].detach().float().norm().cpu()) for name in names}


def build_train_metrics(model, training, local_step, global_step, ae_result, disc_result):
    ae_loss, ae_logs, ae_meta = ae_result
    disc_loss, disc_logs, disc_meta = disc_result
    metrics = {"local_step": local_step, "global_step": global_step,
               "train/ae_loss": float(ae_loss.detach().cpu()), "train/disc_loss": float(disc_loss.detach().cpu()),
               "train/ae_scale": float(ae_meta["scale"].cpu()), "train/ae_latent": float(ae_meta["latent"].cpu()),
               "train/disc_scale": float(disc_meta["scale"].cpu()), "train/disc_latent": float(disc_meta["latent"].cpu())}
    metrics.update(to_scalars(ae_logs)); metrics.update(to_scalars(disc_logs))
    if training.logging.print_parameter_norms:
        metrics.update(parameter_norms(model, training.comparison.parameter_names))
    return metrics


def write_json_line(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(values, sort_keys=True) + "\n")


def progress_values(metrics, optimizers):
    return {"lstep": metrics["local_step"], "gstep": metrics["global_step"],
            "lr": f"{optimizers[0].param_groups[0]['lr']:.2g}",
            "ae": f"{metrics['train/ae_loss']:.3g}", "rec": f"{metrics['train/weighted_pixel_loss']:.3g}",
            "lpips": f"{metrics['train/weighted_lpips_loss']:.3g}", "gan": f"{metrics['train/weighted_g_loss']:.3g}",
            "kl": f"{metrics['train/weighted_kl_loss']:.3g}"}


def write_evaluation(path, metrics, local_step, global_step):
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    line = (f"{timestamp} local_step={local_step} global_step={global_step} samples={metrics['samples']} "
            f"FID={metrics['val/fid']:.6f} "
            f"PSNR={metrics['val/psnr']:.6f} LPIPS={metrics['val/lpips']:.6f} SSIM={metrics['val/ssim']:.6f}\n")
    with path.open("a", encoding="utf-8") as output:
        output.write(line)


def save_image_grid(grid, path):
    array = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((array * 255).clip(0, 255).astype(np.uint8)).save(path)


def log_reconstruction_images(inputs, reconstructions, split, local_step, global_step, config):
    images = {"inputs": inputs, "reconstructions": reconstructions}
    max_images, output_dir = int(config.pytorch_training.logging.max_images), Path(config.pytorch_training.checkpoint.output_dir)
    for name, values in images.items():
        values = values[:max_images].detach().cpu().clamp(-1, 1)
        grid = torchvision.utils.make_grid(values, nrow=4)
        normalized = (grid + 1.0) / 2.0
        filename = f"{name}_local-{local_step:09d}_global-{global_step:09d}.png"
        save_image_grid(normalized, output_dir / "images" / split / filename)


# --- distributed validation -----------------------------------------------------------------------

class ReconstructionMetrics:
    def __init__(self, device):
        self.fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=False, sync_on_compute=True).to(device)
        self.real_features_cached = False
        self.device = device

    def begin(self):
        self.sums = torch.zeros(3, dtype=torch.float64, device=self.device)
        self.samples = torch.zeros((), dtype=torch.long, device=self.device)

    @torch.no_grad()
    def update(self, inputs, reconstructions, perceptual_loss):
        real = torch.clamp(127.5 * inputs + 128.0, 0, 255).to(torch.uint8)
        fake = torch.clamp(127.5 * reconstructions + 128.0, 0, 255).to(torch.uint8)
        if not self.real_features_cached:
            self.fid.update(real, real=True)
        self.fid.update(fake, real=False)
        batch_size = inputs.shape[0]
        mse = (real.float() - fake.float()).pow(2).flatten(1).mean(1)
        psnr = 20.0 * torch.log10(255.0 / torch.sqrt(mse.clamp_min(1e-12)))
        lpips = perceptual_loss(inputs.contiguous(), reconstructions.contiguous()).reshape(batch_size, -1).mean(1)
        ssim = structural_similarity_index_measure(reconstructions.float(), inputs.float(), data_range=2.0, reduction="none")
        self.sums += torch.stack([psnr.sum(), lpips.double().sum(), ssim.double().sum()])
        self.samples += batch_size

    def finalize(self, distributed_context):
        fid = float(self.fid.compute().detach().cpu())
        sums = distributed_context.all_reduce_sum(self.sums)
        samples = distributed_context.all_reduce_sum(self.samples)
        averages = sums / samples.clamp_min(1)
        metrics = {"val/fid": fid, "val/psnr": float(averages[0]), "val/lpips": float(averages[1]),
                   "val/ssim": float(averages[2]), "samples": int(samples)}
        self.real_features_cached = True
        self.fid.reset()
        return metrics


@torch.no_grad()
def run_validation(model, val_loader, metrics, device, distributed_context, config, local_step, global_step):
    distributed_context.barrier()
    if distributed_context.is_main:
        print(f"Starting eval: local_step={local_step} global_step={global_step}", flush=True)
    was_training = model.training
    model.eval(); metrics.begin()
    for batch_index, batch in enumerate(val_loader):
        batch = move_batch_to_device(batch, device)
        outputs = model.compute_validation_outputs(batch)
        metrics.update(outputs["inputs"], outputs["reconstructions"], model.loss.perceptual_loss)
        if batch_index == 0 and distributed_context.is_main and config.pytorch_training.logging.validation_images:
            log_reconstruction_images(
                outputs["inputs"], outputs["reconstructions"], "val", local_step, global_step, config)
    results = metrics.finalize(distributed_context)
    if was_training:
        model.train()
    distributed_context.barrier()
    return results


# --- main train loop and executable ---------------------------------------------------------------

def train(autoencoder, train_model, optimizers, train_loader, train_sampler, val_loader, config, device, resume_state,
          distributed_context):
    training = config.pytorch_training
    output_dir = Path(training.checkpoint.output_dir)
    metrics_path, eval_path = output_dir / training.logging.metrics_filename, output_dir / training.validation.log_filename
    if distributed_context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        if training.logging.overwrite_metrics_file and metrics_path.exists():
            metrics_path.unlink()
    distributed_context.barrier()

    global_step, start_epoch, local_step = resume_state["global_step"], resume_state["epoch"], 0
    max_batches = None if training.max_batches is None else int(training.max_batches)
    accumulation, last_epoch = int(training.accumulate_grad_batches), start_epoch
    validation_interval = int(training.validation.interval_local_steps)
    next_validation_local_step = ((local_step // validation_interval) + 1) * validation_interval
    checkpoint_manager = CheckpointManager(config, resume_state.get("topk")) if distributed_context.is_main else None
    validation_metrics = ReconstructionMetrics(device) if training.validation.enabled else None
    train_model.train()

    for epoch in range(start_epoch, start_epoch + int(training.max_epochs)):
        last_epoch = epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        remaining = len(train_loader) if max_batches is None else min(len(train_loader), max_batches - local_step)
        progress = tqdm(total=remaining, desc="train", dynamic_ncols=True, file=sys.stdout,
                        disable=not distributed_context.is_main, mininterval=0, maxinterval=float("inf"),
                        miniters=int(training.logging.progress_refresh_steps))
        for batch_index, batch in enumerate(train_loader):
            if max_batches is not None and local_step >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            should_step = (local_step + 1) % accumulation == 0 or (max_batches is not None and local_step + 1 == max_batches)
            ae_loss, ae_logs, ae_meta, global_step = optimizer_pass(
                train_model, optimizers, batch, 0, global_step, local_step, accumulation, should_step,
                training.optimizer.zero_grad_set_to_none, distributed_context)
            disc_loss, disc_logs, disc_meta, global_step = optimizer_pass(
                train_model, optimizers, batch, 1, global_step, local_step, accumulation, should_step,
                training.optimizer.zero_grad_set_to_none, distributed_context)
            local_step += 1

            if distributed_context.is_main:
                values = build_train_metrics(autoencoder, training, local_step, global_step,
                                             (ae_loss, ae_logs, ae_meta), (disc_loss, disc_logs, disc_meta))
                progress.set_postfix(progress_values(values, optimizers), refresh=False)
                progress.update(1)
                if local_step % int(training.logging.log_every_n_steps) == 0:
                    write_json_line(metrics_path, values)

            if validation_metrics is not None and local_step >= next_validation_local_step:
                results = run_validation(
                    autoencoder, val_loader, validation_metrics, device, distributed_context,
                    config, local_step, global_step)
                if distributed_context.is_main:
                    write_evaluation(eval_path, results, local_step, global_step)
                    checkpoint_manager.update(results, autoencoder, optimizers, global_step, epoch, local_step)
                next_validation_local_step += validation_interval
            if max_batches is not None and local_step >= max_batches:
                break
        progress.close()
        if max_batches is not None and local_step >= max_batches:
            break

    if validation_metrics is not None and training.validation.run_at_end and local_step < next_validation_local_step:
        results = run_validation(
            autoencoder, val_loader, validation_metrics, device, distributed_context,
            config, local_step, global_step)
        if distributed_context.is_main:
            write_evaluation(eval_path, results, local_step, global_step)
            checkpoint_manager.update(results, autoencoder, optimizers, global_step, last_epoch, local_step)
    elif distributed_context.is_main:
        checkpoint_manager.save_last_checkpoint(autoencoder, optimizers, global_step, last_epoch, local_step)
    distributed_context.barrier()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    training = config.pytorch_training
    distributed_context = DistributedContext(str(training.distributed.backend))
    device = resolve_device(training, distributed_context)
    configure_reproducibility(training.seed, training.deterministic, training.benchmark, distributed_context.rank)
    
    try:
        autoencoder, train_model, optimizers, _ = build_model_and_optimizers(config, device, distributed_context)
        checkpoint = training.checkpoint
        resume_state = {"global_step": 0, "epoch": 0, "topk": []}
        
        if checkpoint.resume_path:
            resume_state = load_checkpoint(
                checkpoint.resume_path, autoencoder, optimizers, device,
                checkpoint.strict_model_load, checkpoint.resume_optimizer
            )
        distributed_context.barrier()

        train_loader, train_sampler, val_loader = build_data_loaders(config, distributed_context)
        
        if distributed_context.is_main:
            output_dir = Path(checkpoint.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(config, output_dir / "resolved_config.yaml")
        
        train(autoencoder, train_model, optimizers, train_loader, train_sampler, val_loader, config, device, resume_state,
              distributed_context)
    finally:
        distributed_context.close()


if __name__ == "__main__":
    main()
