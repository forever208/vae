"""Reusable dual-optimizer training mechanics."""

import contextlib


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
        loss, log_dict, metadata = train_model(batch, optimizer_index, global_step)
        (loss / accumulation).backward()

    if should_step:
        optimizer.step()
        global_step += 1
        
    return loss, log_dict, metadata, global_step
