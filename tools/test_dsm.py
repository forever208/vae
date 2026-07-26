"""Fast checks for DSM filtering and branch selection."""

from unittest import mock

import torch
import torch.nn.functional as F

from models.vae import AutoencoderKL
from utils_DCT import (
    combine_blocks_torch,
    dct_2d_torch_unified,
    idct_2d_torch_unified,
    split_into_blocks_torch,
)


class _Posterior:
    def __init__(self, latent):
        self.latent = latent

    def sample(self):
        return self.latent


class _TinyAutoencoder(AutoencoderKL):
    def __init__(self, p_prior, p_prior_s=0.0, blk_sz=8):
        torch.nn.Module.__init__(self)
        self.image_key = "image"
        self.p_prior = p_prior
        self.p_prior_s = p_prior_s
        self.blk_sz = blk_sz

    def encode(self, inputs):
        return _Posterior(
            F.interpolate(
                inputs[:, :1], scale_factor=0.5, mode="bilinear",
                align_corners=False,
            )
        )

    def decode(self, latents):
        return F.interpolate(
            latents.repeat(1, 3, 1, 1), scale_factor=2,
            mode="bilinear", align_corners=False,
        )


def check_dct_round_trip():
    inputs = torch.randn(2, 3, 16, 16)
    blocks = split_into_blocks_torch(inputs, 8)
    coefficients = dct_2d_torch_unified(blocks, center="none")
    restored_blocks = idct_2d_torch_unified(coefficients, center="none")
    restored = combine_blocks_torch(restored_blocks, 16, 16, 8)
    assert restored.shape == inputs.shape
    assert torch.allclose(restored, inputs, atol=1e-5, rtol=1e-5)


def check_low_pass():
    checkerboard = (
        (torch.arange(16).view(1, 1, 16, 1)
         + torch.arange(16).view(1, 1, 1, 16)) % 2
    ).float() * 2 - 1
    filtered = AutoencoderKL._dsm_low_pass(checkerboard, 12, 8)
    assert filtered.shape == checkerboard.shape
    assert filtered.square().mean() < checkerboard.square().mean()


def check_dsm_branch():
    model = _TinyAutoencoder(p_prior=1.0)
    batch = {"image": torch.randn(2, 16, 16, 3)}
    with mock.patch("models.vae.random.random", return_value=0.0), \
            mock.patch("models.vae.random.choice", return_value=10):
        targets, reconstructions, _, scale, mode = model._training_forward(batch)
    assert mode == "dsm"
    assert scale == 1
    assert targets.shape == reconstructions.shape == (2, 3, 16, 16)


def check_image_branch():
    model = _TinyAutoencoder(p_prior=0.0, p_prior_s=0.0)
    batch = {"image": torch.randn(2, 16, 16, 3)}
    with mock.patch("models.vae.random.random", return_value=1.0):
        targets, reconstructions, _, scale, mode = model._training_forward(batch)
    assert mode == "image"
    assert scale == 1
    assert targets.shape == reconstructions.shape == (2, 3, 16, 16)


def check_low_resolution_image_branch():
    model = _TinyAutoencoder(p_prior=0.0, p_prior_s=1.0)
    batch = {"image": torch.randn(2, 16, 16, 3)}
    with mock.patch(
        "models.vae.random.random", side_effect=(1.0, 0.0)
    ), mock.patch("models.vae.random.choice", return_value=0.5):
        targets, reconstructions, _, scale, mode = model._training_forward(batch)
    assert mode == "image"
    assert scale == 0.5
    assert targets.shape == reconstructions.shape == (2, 3, 8, 8)


def check_invalid_block_size():
    inputs = torch.randn(1, 3, 15, 16)
    try:
        AutoencoderKL._dsm_low_pass(inputs, 8, 8)
    except ValueError as error:
        assert "must divide spatial shape" in str(error)
    else:
        raise AssertionError("DSM accepted a non-divisible spatial shape")


def main():
    check_dct_round_trip()
    check_low_pass()
    check_dsm_branch()
    check_image_branch()
    check_low_resolution_image_branch()
    check_invalid_block_size()
    print("DSM checks: ok")


if __name__ == "__main__":
    main()
