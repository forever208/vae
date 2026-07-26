"""Checkpoint loading, serialization, and top-k retention."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    path, model, optimizers, device, strict_model_load, resume_optimizer
):
    checkpoint = torch_load(path, device)
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError(f"Checkpoint {path} has no model or state_dict entry")
    if state and next(iter(state)).startswith("module."):
        state = {
            key.removeprefix("module."): value for key, value in state.items()
        }
    incompatible = model.load_state_dict(state, strict=bool(strict_model_load))
    if not strict_model_load and (
        incompatible.missing_keys or incompatible.unexpected_keys
    ):
        print(
            "Checkpoint incompatibilities: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    if resume_optimizer:
        states = checkpoint.get("optimizer_states", [])
        if len(states) != len(optimizers):
            raise ValueError(
                f"Expected {len(optimizers)} optimizer states, "
                f"found {len(states)} in {path}"
            )
        for optimizer, optimizer_state in zip(optimizers, states):
            optimizer.load_state_dict(optimizer_state)
    return {
        "global_step": int(checkpoint.get("global_step", 0)),
        "epoch": int(checkpoint.get("epoch", 0)),
        "topk": checkpoint.get("topk", []),
    }


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint(
    path, model, optimizers, global_step, epoch, local_step, config, topk
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = model.state_dict()
    torch.save(
        {
            "format_version": 2,
            "model": model_state,
            "state_dict": model_state,
            "optimizer_states": [
                optimizer.state_dict() for optimizer in optimizers
            ],
            "global_step": global_step,
            "epoch": epoch,
            "local_step": local_step,
            "topk": topk,
            "rng_state": capture_rng_state(),
            "config": OmegaConf.to_container(config, resolve=True),
        },
        path,
    )


class CheckpointManager:
    def __init__(self, config, resume_topk=None):
        cfg = config.pytorch_training.checkpoint
        self.config = config
        self.directory = Path(cfg.output_dir) / cfg.directory
        self.monitor = str(cfg.monitor)
        self.mode = str(cfg.mode)
        self.save_top_k = int(cfg.save_top_k)
        self.save_last = bool(cfg.save_last)
        self.filename_pattern = str(cfg.filename_pattern)
        self.entries = list(resume_topk or [])

    def save_last_checkpoint(
        self, model, optimizers, global_step, epoch, local_step
    ):
        if self.save_last:
            save_checkpoint(
                self.directory / "last.pt",
                model,
                optimizers,
                global_step,
                epoch,
                local_step,
                self.config,
                self.entries,
            )

    def update(
        self, metrics, model, optimizers, global_step, epoch, local_step
    ):
        score = float(metrics[self.monitor])
        candidate = {
            "score": score,
            "global_step": global_step,
            "path": str(
                self.directory
                / self.filename_pattern.format(
                    global_step=global_step,
                    local_step=local_step,
                    epoch=epoch,
                    score=score,
                )
            ),
        }
        should_keep = self.save_top_k < 0 or len(self.entries) < self.save_top_k
        if not should_keep and self.entries:
            worst = (max if self.mode == "min" else min)(
                self.entries, key=lambda item: item["score"]
            )
            should_keep = (
                score < worst["score"]
                if self.mode == "min"
                else score > worst["score"]
            )
            if should_keep:
                Path(worst["path"]).unlink(missing_ok=True)
                self.entries.remove(worst)
        if should_keep:
            self.entries.append(candidate)
            self.entries.sort(
                key=lambda item: item["score"], reverse=self.mode == "max"
            )
            save_checkpoint(
                candidate["path"],
                model,
                optimizers,
                global_step,
                epoch,
                local_step,
                self.config,
                self.entries,
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "topk.json").write_text(
            json.dumps(self.entries, indent=2, sort_keys=True), encoding="utf-8"
        )
        self.save_last_checkpoint(
            model, optimizers, global_step, epoch, local_step
        )
