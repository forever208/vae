"""EQ-VAE model with latent-space equivariance regularization."""

import random

import torch
import torch.nn.functional as F

from models.vae import AutoencoderKL as BaseAutoencoderKL


class AutoencoderKL(BaseAutoencoderKL):
    """VAE trained with matched scale and rotation transforms in latent space."""

    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        anisotropic=False,
        uniform_sample_scale=True,
        **kwargs,
    ):
        super().__init__(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            embed_dim=embed_dim,
            anisotropic=anisotropic,
            uniform_sample_scale=uniform_sample_scale,
            **kwargs,
        )
        self.anisotropic = anisotropic
        # Retained for compatibility with the EQVAE branch configuration. The
        # reference implementation samples from the same fixed discrete grid.
        self.uniform_sample_scale = uniform_sample_scale

    def forward(self, input, scale=1, angle=0):
        posterior = self.encode(input)
        latents = posterior.sample()
        if scale != 1:
            latents = F.interpolate(
                latents,
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
            )
        if angle != 0:
            latents = torch.rot90(latents, k=angle, dims=(-1, -2))
        return self.decode(latents), posterior, latents

    def _sample_scale(self):
        choices = [scale / 32 for scale in range(8, 32)]
        if self.anisotropic:
            return tuple(random.choice(choices) for _ in range(2))
        return random.choice(choices)

    def _training_forward(self, batch):
        inputs = self.prepare_batch_input(batch, self.image_key)

        if random.random() < self.p_prior:
            mode = "latent"
            scale = self._sample_scale()
            angle = random.choice((1, 2, 3))
            reconstructions, posterior, _ = self(
                inputs,
                scale=scale,
                angle=angle,
            )
            inputs = F.interpolate(
                inputs,
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
            )
            inputs = torch.rot90(inputs, k=angle, dims=(-1, -2))
        else:
            mode = "image"
            if random.random() < self.p_prior_s:
                scale = random.choice(
                    [sample / 32 for sample in range(8, 32)]
                )
                inputs = F.interpolate(
                    inputs,
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                scale = 1
            reconstructions, posterior, _ = self(inputs)

        return inputs, reconstructions, posterior, scale, mode

    def compute_train_loss(self, batch, optimizer_idx, global_step):
        inputs, reconstructions, posterior, scale, mode = (
            self._training_forward(batch)
        )
        loss, log_dict = self.loss(
            inputs,
            reconstructions,
            posterior,
            optimizer_idx,
            global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        metadata = {
            "scale": torch.as_tensor(
                scale,
                device=inputs.device,
            ).float().mean(),
            "latent": torch.as_tensor(
                float(mode == "latent"),
                device=inputs.device,
            ),
        }
        return loss, log_dict, metadata

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = {}
        inputs = self.prepare_batch_input(
            batch,
            self.image_key,
        ).to(next(self.parameters()).device)
        if not only_inputs:
            scale = 1 if random.random() < 0.5 else 0.5
            reconstructions, posterior, _ = self(inputs, scale=scale)
            if inputs.shape[1] > 3:
                assert reconstructions.shape[1] > 3
                inputs = self.to_rgb(inputs)
                reconstructions = self.to_rgb(reconstructions)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = reconstructions
        log["inputs"] = inputs
        return log
