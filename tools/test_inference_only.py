"""Checks for lightweight AutoencoderKL inference mode."""

import sys

import torch

from models.vae import AutoencoderKL
from models.vavae import AutoencoderKL as VAVAE


TINY_DDCONFIG = {
    "double_z": True,
    "z_channels": 4,
    "resolution": 8,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 32,
    "ch_mult": (1,),
    "num_res_blocks": 1,
    "attn_resolutions": (),
    "dropout": 0.0,
}


def check_lightweight_construction():
    assert "losses.vae_loss" not in sys.modules
    assert "utils_DCT" not in sys.modules

    model = AutoencoderKL(
        ddconfig=TINY_DDCONFIG,
        lossconfig=None,
        embed_dim=4,
        inference_only=True,
    ).eval()
    assert model.encoder is not None
    assert model.decoder is not None
    assert model.loss is None
    assert "losses.vae_loss" not in sys.modules
    assert "utils_DCT" not in sys.modules

    inputs = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        reconstructions, posterior, latents = model(inputs)
    assert posterior.mean.shape == latents.shape == (2, 4, 8, 8)
    assert reconstructions.shape == inputs.shape


def check_training_guards():
    model = AutoencoderKL(
        ddconfig=TINY_DDCONFIG,
        lossconfig=None,
        embed_dim=4,
        inference_only=True,
    )
    batch = {"image": torch.randn(1, 8, 8, 3)}
    for operation in (
        lambda: model._training_forward(batch),
        lambda: model.compute_train_loss(batch, 0, 0),
        lambda: model.apply_dsm(
            torch.randn(1, 3, 8, 8),
            torch.randn(1, 4, 8, 8),
            8,
        ),
    ):
        try:
            operation()
        except RuntimeError as error:
            assert "inference_only=True" in str(error)
        else:
            raise AssertionError("Inference-only model accepted a training operation")
    assert "utils_DCT" not in sys.modules


def check_mode_validation():
    try:
        AutoencoderKL(
            ddconfig=TINY_DDCONFIG,
            lossconfig=None,
            embed_dim=4,
            encoder_only=True,
            inference_only=True,
        )
    except ValueError as error:
        assert "cannot both be enabled" in str(error)
    else:
        raise AssertionError("Conflicting lightweight modes were accepted")


def check_vavae_modes():
    model = VAVAE(
        ddconfig=TINY_DDCONFIG,
        embed_dim=4,
    ).eval()
    inputs = torch.randn(1, 3, 8, 8)
    reconstruction = model.reconstruct(inputs)
    assert reconstruction.shape == inputs.shape
    assert model.decoder is not None
    assert model.loss is None
    assert model.inference_only

    encoder = VAVAE(
        ddconfig=TINY_DDCONFIG,
        embed_dim=4,
        encoder_only=True,
        inference_only=False,
    )
    assert encoder.decoder is None
    assert encoder.loss is None
    assert not encoder.inference_only


def main():
    check_lightweight_construction()
    check_training_guards()
    check_mode_validation()
    check_vavae_modes()
    print("Inference-only checks: ok")


if __name__ == "__main__":
    main()
