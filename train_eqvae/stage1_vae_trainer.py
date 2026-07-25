"""EQ-VAE stage-one CLI, configuration, and training lifecycle."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from training.builders import build_data_loaders, build_model_and_optimizers
from training.checkpoint import load_checkpoint
from training.distributed import (
    DistributedContext,
    configure_reproducibility,
    resolve_device,
)
from training.loop import train


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch trainer for EQ-VAE")
    parser.add_argument(
        "--config",
        default="configs/eqvae_config.yaml",
        help="Complete YAML training configuration",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    training = config.pytorch_training
    distributed_context = DistributedContext(
        str(training.distributed.backend)
    )
    device = resolve_device(training, distributed_context)
    configure_reproducibility(
        training.seed,
        training.deterministic,
        training.benchmark,
        distributed_context.rank,
    )

    try:
        autoencoder, train_model, optimizers, _ = build_model_and_optimizers(
            config, device, distributed_context
        )
        checkpoint = training.checkpoint
        resume_state = {"global_step": 0, "epoch": 0, "topk": []}

        if checkpoint.resume_path:
            resume_state = load_checkpoint(
                checkpoint.resume_path,
                autoencoder,
                optimizers,
                device,
                checkpoint.strict_model_load,
                checkpoint.resume_optimizer,
            )
        distributed_context.barrier()

        train_loader, train_sampler, val_loader = build_data_loaders(
            config, distributed_context
        )

        if distributed_context.is_main:
            output_dir = Path(checkpoint.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(config, output_dir / "resolved_config.yaml")

        train(
            autoencoder,
            train_model,
            optimizers,
            train_loader,
            train_sampler,
            val_loader,
            config,
            device,
            resume_state,
            distributed_context,
        )
    finally:
        distributed_context.close()


if __name__ == "__main__":
    main()
