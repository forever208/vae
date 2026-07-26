"""EQ-VAE stage-one CLI, configuration, and training lifecycle."""

import argparse
import sys
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

from training.builders import build_data_loaders, build_model_and_optimizers, move_batch_to_device
from training.checkpoint import CheckpointManager, load_checkpoint
from training.distributed import DistributedContext, configure_reproducibility, resolve_device
from training.logging import build_train_metrics, progress_values, write_evaluation
from training.loop import optimizer_pass
from training.metrics import ReconstructionMetrics, run_validation


def parse_args():
    """Parse the training configuration and common CLI overrides."""
    parser = argparse.ArgumentParser(description="PyTorch trainer for EQ-VAE")
    parser.add_argument("--config", default="configs/eqvae_config.yaml", help="Complete YAML training configuration")
    parser.add_argument("--output_dir", required=True, help="Directory for checkpoints, logs, and validation outputs")
    parser.add_argument("--resume_path", required=True, help="Checkpoint path to resume from; pass an empty string to disable")
    parser.add_argument("--batch_size", type=int, required=True, help="Per-device training batch size")
    parser.add_argument("--max_epochs", type=int, required=True, help="Maximum number of training epochs")
    parser.add_argument("--eval_interval_steps", type=int, required=True, help="Validation interval in local training steps")
    return parser.parse_args()


def main():
    """Initialize the environment and run training, validation, and checkpointing."""
    args = parse_args()
    config = OmegaConf.load(args.config)
    # Command-line arguments override the corresponding YAML settings.
    config.pytorch_training.checkpoint.output_dir = args.output_dir
    config.pytorch_training.checkpoint.resume_path = args.resume_path
    config.data.batch_size = args.batch_size
    config.pytorch_training.max_epochs = args.max_epochs
    config.pytorch_training.validation.eval_interval_steps = args.eval_interval_steps
    training = config.pytorch_training
    # Initialize distributed execution, the compute device, and reproducibility.
    distributed_context = DistributedContext(str(training.distributed.backend))
    device = resolve_device(training, distributed_context)
    configure_reproducibility(training.seed, training.deterministic, training.benchmark, distributed_context.rank,)

    try:
        # Build the model and optimizers, then restore training state if requested.
        autoencoder, train_model, optimizers, _ = build_model_and_optimizers(config, device, distributed_context)
        checkpoint = training.checkpoint
        resume_state = {"global_step": 0, "epoch": 0, "topk": []}

        if checkpoint.resume_path:
            resume_state = load_checkpoint(
                checkpoint.resume_path, autoencoder, optimizers, device,
                checkpoint.strict_model_load, checkpoint.resume_optimizer,
            )
        distributed_context.barrier()

        # Each process builds its loaders; the main process persists run metadata.
        train_loader, train_sampler, val_loader = build_data_loaders(config, distributed_context)

        if distributed_context.is_main:
            per_device_batch_size = int(config.data.batch_size)
            accumulation = int(training.accumulate_grad_batches)
            global_batch_size = (per_device_batch_size * distributed_context.world_size * accumulation)
            print(f"global batch size={global_batch_size}, world size={distributed_context.world_size}")
            print(f"train dataset load path: {config.data.train.params.train_dir}")
            print(f"validation dataset load path: {config.data.validation.params.val_dir}")
            print(f"checkpoint load path: {Path(checkpoint.resume_path).expanduser().resolve()}")
            output_dir = Path(checkpoint.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(config, output_dir / "resolved_config.yaml")

        output_dir = Path(training.checkpoint.output_dir)
        eval_path = output_dir / training.validation.log_filename
        if distributed_context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
        distributed_context.barrier()

        global_step = resume_state["global_step"]
        start_epoch = resume_state["epoch"]
        local_step = 0
        max_batches = None if training.max_batches is None else int(training.max_batches)
        accumulation = int(training.accumulate_grad_batches)
        last_epoch = start_epoch
        validation_interval = int(training.validation.eval_interval_steps)
        progress_refresh_steps = int(training.logging.progress_refresh_steps)
        next_validation_local_step = ((local_step // validation_interval) + 1) * validation_interval
        checkpoint_manager = CheckpointManager(config, resume_state.get("topk")) if distributed_context.is_main else None
        validation_metrics = ReconstructionMetrics(device) if training.validation.enabled else None
        train_model.train()

        # Update the autoencoder and discriminator with gradient accumulation.
        for epoch in range(start_epoch, start_epoch + int(training.max_epochs)):
            last_epoch = epoch
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            remaining = len(train_loader) if max_batches is None else min(len(train_loader), max_batches - local_step)
            progress = tqdm(
                total=remaining, desc="train", dynamic_ncols=True, file=sys.stdout,
                disable=not distributed_context.is_main, mininterval=float("inf"),
                maxinterval=float("inf"), miniters=1, delay=float("inf"),
            )
            for batch_index, batch in enumerate(train_loader):
                if max_batches is not None and local_step >= max_batches:
                    break
                batch = move_batch_to_device(batch, device)
                should_step = (local_step + 1) % accumulation == 0 or (
                    max_batches is not None and local_step + 1 == max_batches
                )
                ae_loss, ae_logs, ae_meta, global_step = optimizer_pass(
                    train_model, optimizers, batch, 0, global_step, local_step,
                    accumulation, should_step, training.optimizer.zero_grad_set_to_none, distributed_context,
                )
                disc_loss, disc_logs, disc_meta, global_step = optimizer_pass(
                    train_model, optimizers, batch, 1, global_step, local_step,
                    accumulation, should_step, training.optimizer.zero_grad_set_to_none, distributed_context,
                )
                local_step += 1  # update both VAE and GAN, then local_step += 1

                if distributed_context.is_main:
                    # Only the main process updates the progress display.
                    values = build_train_metrics(
                        autoencoder, training, local_step, global_step,
                        (ae_loss, ae_logs, ae_meta),
                        (disc_loss, disc_logs, disc_meta),
                    )
                    progress.set_postfix(progress_values(values, optimizers), refresh=False)
                    progress.update(1)
                    if local_step % progress_refresh_steps == 0:
                        progress.refresh()

                # Validate periodically and update ranked checkpoints.
                if validation_metrics is not None and local_step >= next_validation_local_step:
                    results = run_validation(
                        autoencoder, val_loader, validation_metrics, device,
                        distributed_context, config, local_step, global_step,
                    )
                    if distributed_context.is_main:
                        write_evaluation(eval_path, results, local_step, global_step)
                        checkpoint_manager.update(results, autoencoder, optimizers, global_step, epoch, local_step,)
                    next_validation_local_step += validation_interval
                if max_batches is not None and local_step >= max_batches:
                    break

            progress.close()
            if max_batches is not None and local_step >= max_batches:
                break
        
        # save most recent checkpoint
        if distributed_context.is_main:
            checkpoint_manager.save_last_checkpoint(autoencoder, optimizers, global_step, last_epoch, local_step)
        distributed_context.barrier()

    finally:
        # Always release distributed communication resources.
        distributed_context.close()


if __name__ == "__main__":
    main()
