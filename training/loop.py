"""Dual-optimizer training mechanics and epoch loop."""

import contextlib
import sys
from pathlib import Path

from tqdm import tqdm

from .builders import move_batch_to_device
from .checkpoint import CheckpointManager
from .logging import (
    build_train_metrics,
    progress_values,
    write_evaluation,
    write_json_line,
)
from .metrics import ReconstructionMetrics, run_validation


@contextlib.contextmanager
def activate_optimizer(optimizers, active_index):
    requires_grad = {}
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter not in requires_grad:
                    requires_grad[parameter] = parameter.requires_grad
                    parameter.requires_grad = False
    for group in optimizers[active_index].param_groups:
        for parameter in group["params"]:
            parameter.requires_grad = requires_grad[parameter]
    try:
        yield
    finally:
        for parameter, original in requires_grad.items():
            parameter.requires_grad = original


def optimizer_pass(
    train_model,
    optimizers,
    batch,
    optimizer_index,
    global_step,
    local_step,
    accumulation,
    should_step,
    zero_grad_set_to_none,
    distributed_context,
):
    optimizer = optimizers[optimizer_index]
    if local_step % accumulation == 0:
        optimizer.zero_grad(set_to_none=bool(zero_grad_set_to_none))
    sync_context = (
        contextlib.nullcontext()
        if should_step or not distributed_context.distributed
        else train_model.no_sync()
    )
    with sync_context, activate_optimizer(optimizers, optimizer_index):
        loss, log_dict, metadata = train_model(
            batch, optimizer_index, global_step
        )
        (loss / accumulation).backward()
    if should_step:
        optimizer.step()
        global_step += 1
    return loss, log_dict, metadata, global_step


def train(
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
):
    training = config.pytorch_training
    output_dir = Path(training.checkpoint.output_dir)
    metrics_path = output_dir / training.logging.metrics_filename
    eval_path = output_dir / training.validation.log_filename
    if distributed_context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        if training.logging.overwrite_metrics_file and metrics_path.exists():
            metrics_path.unlink()
    distributed_context.barrier()

    global_step = resume_state["global_step"]
    start_epoch = resume_state["epoch"]
    local_step = 0
    max_batches = (
        None if training.max_batches is None else int(training.max_batches)
    )
    accumulation = int(training.accumulate_grad_batches)
    last_epoch = start_epoch
    validation_interval = int(training.validation.interval_local_steps)
    next_validation_local_step = (
        (local_step // validation_interval) + 1
    ) * validation_interval
    checkpoint_manager = (
        CheckpointManager(config, resume_state.get("topk"))
        if distributed_context.is_main
        else None
    )
    validation_metrics = (
        ReconstructionMetrics(device) if training.validation.enabled else None
    )
    train_model.train()

    for epoch in range(start_epoch, start_epoch + int(training.max_epochs)):
        last_epoch = epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        remaining = (
            len(train_loader)
            if max_batches is None
            else min(len(train_loader), max_batches - local_step)
        )
        progress = tqdm(
            total=remaining,
            desc="train",
            dynamic_ncols=True,
            file=sys.stdout,
            disable=not distributed_context.is_main,
            mininterval=0,
            maxinterval=float("inf"),
            miniters=int(training.logging.progress_refresh_steps),
        )
        for batch_index, batch in enumerate(train_loader):
            if max_batches is not None and local_step >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            should_step = (local_step + 1) % accumulation == 0 or (
                max_batches is not None and local_step + 1 == max_batches
            )
            ae_loss, ae_logs, ae_meta, global_step = optimizer_pass(
                train_model,
                optimizers,
                batch,
                0,
                global_step,
                local_step,
                accumulation,
                should_step,
                training.optimizer.zero_grad_set_to_none,
                distributed_context,
            )
            disc_loss, disc_logs, disc_meta, global_step = optimizer_pass(
                train_model,
                optimizers,
                batch,
                1,
                global_step,
                local_step,
                accumulation,
                should_step,
                training.optimizer.zero_grad_set_to_none,
                distributed_context,
            )
            local_step += 1

            if distributed_context.is_main:
                values = build_train_metrics(
                    autoencoder,
                    training,
                    local_step,
                    global_step,
                    (ae_loss, ae_logs, ae_meta),
                    (disc_loss, disc_logs, disc_meta),
                )
                progress.set_postfix(
                    progress_values(values, optimizers), refresh=False
                )
                progress.update(1)
                if (
                    local_step
                    % int(training.logging.log_every_n_steps)
                    == 0
                ):
                    write_json_line(metrics_path, values)

            if (
                validation_metrics is not None
                and local_step >= next_validation_local_step
            ):
                results = run_validation(
                    autoencoder,
                    val_loader,
                    validation_metrics,
                    device,
                    distributed_context,
                    config,
                    local_step,
                    global_step,
                )
                if distributed_context.is_main:
                    write_evaluation(
                        eval_path, results, local_step, global_step
                    )
                    checkpoint_manager.update(
                        results,
                        autoencoder,
                        optimizers,
                        global_step,
                        epoch,
                        local_step,
                    )
                next_validation_local_step += validation_interval
            if max_batches is not None and local_step >= max_batches:
                break
        progress.close()
        if max_batches is not None and local_step >= max_batches:
            break

    if (
        validation_metrics is not None
        and training.validation.run_at_end
        and local_step < next_validation_local_step
    ):
        results = run_validation(
            autoencoder,
            val_loader,
            validation_metrics,
            device,
            distributed_context,
            config,
            local_step,
            global_step,
        )
        if distributed_context.is_main:
            write_evaluation(eval_path, results, local_step, global_step)
            checkpoint_manager.update(
                results,
                autoencoder,
                optimizers,
                global_step,
                last_epoch,
                local_step,
            )
    elif distributed_context.is_main:
        checkpoint_manager.save_last_checkpoint(
            autoencoder,
            optimizers,
            global_step,
            last_epoch,
            local_step,
        )
    distributed_context.barrier()
