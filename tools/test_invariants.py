"""Fast invariant checks for the EQ-VAE training package."""

import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

from core.config import instantiate_from_config
from data.datasets import DatasetTrain, DatasetVal
from losses import lpips as lpips_module
from losses.vae_loss import LPIPSWithDiscriminator
from models.vae import AutoencoderKL
from training.checkpoint import save_checkpoint, torch_load


def check_import_contract():
    assert AutoencoderKL.__name__ == "AutoencoderKL"
    assert LPIPSWithDiscriminator.__name__ == "LPIPSWithDiscriminator"
    assert DatasetTrain.__name__ == "DatasetTrain"
    assert DatasetVal.__name__ == "DatasetVal"
    assert callable(instantiate_from_config)


def check_checkpoint_contract():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = OmegaConf.create({"test": True})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        save_checkpoint(path, model, [optimizer], 7, 2, 5, config, [])
        checkpoint = torch_load(path, "cpu")
    assert checkpoint["format_version"] == 2
    assert checkpoint["global_step"] == 7
    assert checkpoint["local_step"] == 5
    assert checkpoint["model"].keys() == checkpoint["state_dict"].keys()
    for key in checkpoint["model"]:
        assert torch.equal(checkpoint["model"][key], checkpoint["state_dict"][key])


def check_lpips_key_contract():
    original_vgg16 = lpips_module.models.vgg16

    class TinyVGG:
        features = torch.nn.Sequential(
            *(torch.nn.Conv2d(1, 1, 1) for _ in range(30))
        )

    try:
        lpips_module.models.vgg16 = lambda pretrained: TinyVGG()
        network = lpips_module.vgg16(pretrained=False)
    finally:
        lpips_module.models.vgg16 = original_vgg16

    keys = set(network.state_dict())
    assert "slice2.5.weight" in keys
    assert "slice2.1.weight" not in keys
    assert "slice5.28.weight" in keys


def main():
    check_import_contract()
    check_checkpoint_contract()
    check_lpips_key_contract()
    print("EQ-VAE invariants: ok")


if __name__ == "__main__":
    main()
