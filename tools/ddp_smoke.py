"""CPU-only two-process smoke test for the VAE DDP optimizer path."""

import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.builders import DistributedTrainStep
from training.distributed import DistributedContext
from training.loop import optimizer_pass


class ToyAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ae = torch.nn.Linear(2, 2)
        self.discriminator = torch.nn.Linear(2, 1)

    def compute_train_loss(self, batch, optimizer_index, global_step):
        layer = self.ae if optimizer_index == 0 else self.discriminator
        loss = layer(batch["image"]).square().mean()
        metadata = {"scale": torch.tensor(1.0), "dsm": torch.tensor(0.0)}
        return loss, {f"loss/{optimizer_index}": loss.detach()}, metadata


def main():
    context = DistributedContext("gloo")
    torch.manual_seed(7)
    model = ToyAutoencoder()
    optimizers = [torch.optim.Adam(model.ae.parameters(), 1e-3), torch.optim.Adam(model.discriminator.parameters(), 1e-3)]
    train_model = DDP(DistributedTrainStep(model), find_unused_parameters=True)
    batch, global_step = {"image": torch.full((2, 2), float(context.rank + 1))}, 0
    for optimizer_index in (0, 1):
        *_, global_step = optimizer_pass(train_model, optimizers, batch, optimizer_index, global_step, 0, 1, True, True, context)
    assert global_step == 2
    for parameter in model.parameters():
        reference = parameter.detach().clone()
        torch.distributed.broadcast(reference, src=0)
        assert torch.allclose(parameter, reference)
    if context.is_main:
        print("DDP smoke: ok")
    context.close()


if __name__ == "__main__":
    main()
