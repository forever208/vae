"""Data, model, optimizer, and batch construction helpers."""

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from core.config import instantiate_from_config


def calculate_learning_rate(config, world_size):
    base_lr = float(config.model.base_learning_rate)
    training = config.pytorch_training
    if not training.scale_learning_rate:
        return base_lr
    return (
        base_lr
        * int(config.data.batch_size)
        * world_size
        * int(training.accumulate_grad_batches)
    )


def build_loader(
    dataset_config, config, distributed_context, dataloader_config, shuffle
):
    dataset = instantiate_from_config(dataset_config)
    sampler = (
        DistributedSampler(
            dataset,
            distributed_context.world_size,
            distributed_context.rank,
            shuffle=shuffle,
            seed=int(config.pytorch_training.seed),
            drop_last=bool(dataloader_config.drop_last),
        )
        if distributed_context.distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.data.batch_size),
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=int(dataloader_config.num_workers),
        pin_memory=bool(dataloader_config.pin_memory),
        drop_last=bool(dataloader_config.drop_last) if sampler is None else False,
    )
    return loader, sampler


def build_data_loaders(config, distributed_context):
    training = config.pytorch_training
    train_loader, train_sampler = build_loader(
        config.data.train,
        config,
        distributed_context,
        training.dataloader,
        bool(training.dataloader.shuffle),
    )
    val_loader, _ = build_loader(
        config.data.validation,
        config,
        distributed_context,
        training.validation.dataloader,
        False,
    )
    return train_loader, train_sampler, val_loader


class DistributedTrainStep(torch.nn.Module):
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, batch, optimizer_index, global_step):
        return self.autoencoder.compute_train_loss(
            batch, optimizer_index, global_step
        )


def build_model_and_optimizers(config, device, distributed_context):
    training = config.pytorch_training
    autoencoder = instantiate_from_config(config.model).to(device)
    learning_rate = calculate_learning_rate(config, distributed_context.world_size)
    optimizers = autoencoder.build_optimizers(
        learning_rate,
        betas=tuple(float(beta) for beta in training.optimizer.betas),
    )
    train_model = DistributedTrainStep(autoencoder).to(device)
    if distributed_context.distributed:
        train_model = DDP(
            train_model,
            device_ids=[distributed_context.local_rank],
            output_device=distributed_context.local_rank,
            find_unused_parameters=bool(
                training.distributed.find_unused_parameters
            ),
            broadcast_buffers=bool(training.distributed.broadcast_buffers),
        )
    return autoencoder, train_model, optimizers, learning_rate


def move_batch_to_device(batch, device):
    return {
        key: (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }
