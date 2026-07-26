"""Distributed process lifecycle, device selection, and reproducibility."""

import os
import random

import numpy as np
import torch
import torch.distributed as dist


class DistributedContext:
    def __init__(self, backend):
        self.rank, self.local_rank, self.world_size = (
            int(os.environ.get(name, default))
            for name, default in (("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1))
        )
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
            dist.destroy_process_group()


def configure_reproducibility(seed, deterministic, benchmark, rank):
    process_seed = int(seed) + int(rank)
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(benchmark)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(training, distributed_context):
    use_local_cuda = distributed_context.distributed and torch.cuda.is_available()
    device = (
        torch.device("cuda", distributed_context.local_rank)
        if use_local_cuda
        else torch.device(training.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    return device
