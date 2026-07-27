"""VAE model with DSM regularization."""

import random
import torch
import torch.nn.functional as F

from losses.vae_loss import LPIPSWithDiscriminator
from models.backbone import Decoder, Encoder
from models.distributions import DiagonalGaussianDistribution
from utils_DCT import (
    combine_blocks_torch,
    dct_2d_torch_unified,
    idct_2d_torch_unified,
    split_into_blocks_torch,
)


class AutoencoderKL(torch.nn.Module):
    def __init__(
        self, ddconfig, lossconfig, embed_dim, anisotropic=False,
        uniform_sample_scale=True, ckpt_path=None, p_prior=0.5, p_prior_s=0.25,
        blk_sz=8, ignore_keys=None, image_key="image", colorize_nlabels=None,
        monitor=None, encoder_only=False,
    ):
        super().__init__()
        ignore_keys = ignore_keys or []
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        if encoder_only:
            self.decoder = None
            self.loss = None
        else:
            self.decoder = Decoder(**ddconfig)
            self.loss = LPIPSWithDiscriminator(**lossconfig)
        # Retain these arguments so historical EQ-VAE configs still load.
        del anisotropic, uniform_sample_scale
        self.p_prior = p_prior
        self.p_prior_s = p_prior_s
        self.blk_sz = blk_sz
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = (
            None
            if encoder_only
            else torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        )
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert isinstance(colorize_nlabels, int)
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=None):
        ignore_keys = ignore_keys or []
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("model", checkpoint.get("state_dict"))
        if state is None:
            raise KeyError(f"Checkpoint {path} has no model or state_dict entry")
        for key in list(state):
            if any(key.startswith(prefix) for prefix in ignore_keys):
                print(f"Deleting key {key} from state_dict.")
                del state[key]
        self.load_state_dict(state, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        return DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)))

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, input):
        posterior = self.encode(input)
        z = posterior.sample()
        return self.decode(z), posterior, z

    @staticmethod
    def _dsm_low_pass(x, dsm_mask, blk_sz):
        height, width = x.shape[-2:]
        if height % blk_sz != 0 or width % blk_sz != 0:
            raise ValueError(f"DSM block size {blk_sz} must divide spatial shape {height}x{width}")

        blocks = split_into_blocks_torch(x, blk_sz)
        coefficients = dct_2d_torch_unified(blocks, center="none")

        max_sum = 2 * (blk_sz - 1)
        threshold = max_sum - (dsm_mask - 1)
        u = torch.arange(blk_sz, device=x.device).view(blk_sz, 1)
        v = torch.arange(blk_sz, device=x.device).view(1, blk_sz)
        high_frequency_mask = (u + v) >= threshold
        coefficients[..., high_frequency_mask] = 0.0

        blocks = idct_2d_torch_unified(coefficients, center="none")
        return combine_blocks_torch(blocks, height, width, blk_sz)

    def apply_dsm(self, inputs, latents, dsm_mask):
        return (
            self._dsm_low_pass(inputs, dsm_mask, self.blk_sz),
            self._dsm_low_pass(latents, dsm_mask, self.blk_sz),
        )

    def prepare_batch_input(self, batch, key):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        return x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()

    def _training_forward(self, batch):
        inputs = self.prepare_batch_input(batch, self.image_key)

        # DSM regularization
        if random.random() < self.p_prior:
            mode = "dsm"
            scale = 1
            dsm_mask = random.choice((8, 10, 12))  # DSM mask size under blk_sz=8
            posterior = self.encode(inputs)
            latents = posterior.sample()
            inputs, latents = self.apply_dsm(inputs, latents, dsm_mask)
            reconstructions = self.decode(latents)

        # SDVAE
        else:
            mode = "image"
            if random.random() < self.p_prior_s:
                scale = random.choice([s / 32 for s in range(8, 32)])
                inputs = F.interpolate(inputs, scale_factor=scale, mode="bilinear", align_corners=False)
            else:
                scale = 1
            reconstructions, posterior, _ = self(inputs)
            
        return inputs, reconstructions, posterior, scale, mode

    def compute_train_loss(self, batch, optimizer_idx, global_step):
        inputs, reconstructions, posterior, scale, mode = self._training_forward(batch)
        lpips_scale = 0.5 if mode == "dsm" else 1.0
        loss, log_dict = self.loss(
            inputs, reconstructions, posterior, optimizer_idx, global_step,
            last_layer=self.get_last_layer(), split="train", lpips_scale=lpips_scale,
        )
        metadata = {
            "scale": torch.as_tensor(scale, device=inputs.device).float().mean(),
            "dsm": torch.as_tensor(float(mode == "dsm"), device=inputs.device),
        }
        return loss, log_dict, metadata

    @torch.no_grad()
    def compute_validation_outputs(self, batch):
        inputs = self.prepare_batch_input(batch, self.image_key)
        reconstructions, _, _ = self(inputs)
        return {"inputs": inputs.detach(), "reconstructions": reconstructions.detach()}

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = {}
        x = self.prepare_batch_input(batch, self.image_key).to(next(self.parameters()).device)
        if not only_inputs:
            xrec, posterior, _ = self(x)
            if x.shape[1] > 3:
                assert xrec.shape[1] > 3
                x, xrec = self.to_rgb(x), self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        return 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        super().__init__()
        self.vq_interface = vq_interface

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        return (x, None, [None, None, None]) if self.vq_interface else x

    def forward(self, x, *args, **kwargs):
        return x

