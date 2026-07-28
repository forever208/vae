"""Fast checks for EQ-VAE transforms and config-driven integration."""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from core.config import get_obj_from_str
from models.eqvae import AutoencoderKL
from models.vae import AutoencoderKL as BaseAutoencoderKL
from training.logging import build_train_metrics


class _Posterior:
    def __init__(self, latent):
        self.latent = latent

    def sample(self):
        return self.latent


class _TinyAutoencoder(AutoencoderKL):
    def __init__(self, p_prior, p_prior_s=0.0, anisotropic=False):
        torch.nn.Module.__init__(self)
        self.image_key = "image"
        self.p_prior = p_prior
        self.p_prior_s = p_prior_s
        self.anisotropic = anisotropic

    def encode(self, inputs):
        return _Posterior(
            F.interpolate(
                inputs[:, :1],
                scale_factor=0.5,
                mode="bilinear",
                align_corners=False,
            )
        )

    def decode(self, latents):
        return F.interpolate(
            latents.repeat(1, 3, 1, 1),
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )


def check_forward_transform():
    model = _TinyAutoencoder(p_prior=1.0)
    inputs = torch.randn(2, 3, 16, 16)
    reconstructions, posterior, latents = model(
        inputs,
        scale=0.5,
        angle=1,
    )
    expected = F.interpolate(
        posterior.sample(),
        scale_factor=0.5,
        mode="bilinear",
        align_corners=False,
    )
    expected = torch.rot90(expected, k=1, dims=(-1, -2))
    assert torch.equal(latents, expected)
    assert reconstructions.shape == (2, 3, 8, 8)


def check_isotropic_eqvae_branch():
    model = _TinyAutoencoder(p_prior=1.0)
    batch = {"image": torch.randn(2, 16, 16, 3)}
    with mock.patch(
        "models.eqvae.random.random",
        return_value=0.0,
    ), mock.patch(
        "models.eqvae.random.choice",
        side_effect=(0.5, 1),
    ):
        targets, reconstructions, _, scale, mode = model._training_forward(
            batch
        )
    assert mode == "latent"
    assert scale == 0.5
    assert targets.shape == reconstructions.shape == (2, 3, 8, 8)


def check_anisotropic_eqvae_branch():
    model = _TinyAutoencoder(
        p_prior=1.0,
        anisotropic=True,
    )
    batch = {"image": torch.randn(2, 16, 16, 3)}
    with mock.patch(
        "models.eqvae.random.random",
        return_value=0.0,
    ), mock.patch(
        "models.eqvae.random.choice",
        side_effect=(0.5, 0.75, 1),
    ):
        targets, reconstructions, _, scale, mode = model._training_forward(
            batch
        )
    assert mode == "latent"
    assert scale == (0.5, 0.75)
    assert targets.shape == reconstructions.shape == (2, 3, 12, 8)


def check_image_branches():
    batch = {"image": torch.randn(2, 16, 16, 3)}
    model = _TinyAutoencoder(p_prior=0.0, p_prior_s=0.0)
    with mock.patch("models.eqvae.random.random", return_value=1.0):
        targets, reconstructions, _, scale, mode = model._training_forward(
            batch
        )
    assert mode == "image"
    assert scale == 1
    assert targets.shape == reconstructions.shape == (2, 3, 16, 16)

    model = _TinyAutoencoder(p_prior=0.0, p_prior_s=1.0)
    with mock.patch(
        "models.eqvae.random.random",
        side_effect=(1.0, 0.0),
    ), mock.patch(
        "models.eqvae.random.choice",
        return_value=0.5,
    ):
        targets, reconstructions, _, scale, mode = model._training_forward(
            batch
        )
    assert mode == "image"
    assert scale == 0.5
    assert targets.shape == reconstructions.shape == (2, 3, 8, 8)


def check_config_target():
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs/eqvae_config_imgnet256.yaml")
    model_class = get_obj_from_str(config.model.target)
    assert model_class is AutoencoderKL
    assert issubclass(model_class, BaseAutoencoderKL)


def check_metadata_logging():
    training = SimpleNamespace(
        logging=SimpleNamespace(print_parameter_norms=False)
    )
    loss = torch.tensor(1.0)
    logs = {"train/rec_loss": torch.tensor(0.5)}
    eqvae_result = (
        loss,
        logs,
        {"scale": torch.tensor(0.5), "latent": torch.tensor(1.0)},
    )
    metrics = build_train_metrics(
        None,
        training,
        1,
        2,
        eqvae_result,
        eqvae_result,
    )
    assert metrics["train/ae_scale"] == 0.5
    assert metrics["train/ae_latent"] == 1.0
    assert metrics["train/disc_latent"] == 1.0

    dsm_result = (
        loss,
        logs,
        {"scale": torch.tensor(1.0), "dsm": torch.tensor(1.0)},
    )
    metrics = build_train_metrics(
        None,
        training,
        1,
        2,
        dsm_result,
        dsm_result,
    )
    assert metrics["train/ae_dsm"] == 1.0
    assert metrics["train/disc_dsm"] == 1.0


def main():
    check_forward_transform()
    check_isotropic_eqvae_branch()
    check_anisotropic_eqvae_branch()
    check_image_branches()
    check_config_target()
    check_metadata_logging()
    print("EQ-VAE checks: ok")


if __name__ == "__main__":
    main()
