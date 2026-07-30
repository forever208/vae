"""Inference-only Vision Foundation Model Aligned VAE."""

import torch

from models.vae import AutoencoderKL as BaseAutoencoderKL


class AutoencoderKL(BaseAutoencoderKL):
    """VA-VAE f16d32 inference model.

    The released VA-VAE uses the same LDM/VQGAN-KL backbone as the base
    autoencoder. Its architecture is selected through ``ddconfig``; this
    subclass only supplies inference semantics and never constructs a
    perceptual loss or discriminator.
    """

    def __init__(
        self,
        ddconfig,
        embed_dim=32,
        lossconfig=None,
        encoder_only=False,
        inference_only=True,
        **kwargs,
    ):
        if not inference_only and not encoder_only:
            raise ValueError("VA-VAE only supports inference or encoder-only mode")
        super().__init__(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            embed_dim=embed_dim,
            encoder_only=encoder_only,
            inference_only=inference_only and not encoder_only,
            **kwargs,
        )

    @torch.inference_mode()
    def reconstruct(self, inputs, sample_posterior=False):
        """Reconstruct normalized NCHW images in the [-1, 1] range."""
        if self.decoder is None:
            raise RuntimeError("reconstruct requires a decoder")
        posterior = self.encode(inputs)
        latents = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(latents)
