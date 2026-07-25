"""EQ-VAE model implementation."""

import random
import torch
import torch.nn.functional as F

try:
    from core.config import instantiate_from_config
    from models.backbone import Decoder, Encoder
    from models.distributions import DiagonalGaussianDistribution
except ModuleNotFoundError:
    from train_eqvae.core.config import instantiate_from_config
    from train_eqvae.models.backbone import Decoder, Encoder
    from train_eqvae.models.distributions import DiagonalGaussianDistribution


class AutoencoderKL(torch.nn.Module):
    def __init__(
        self, ddconfig, lossconfig, embed_dim, anisotropic=False,
        uniform_sample_scale=True, ckpt_path=None, p_prior=0.5, p_prior_s=0.25,
        ignore_keys=None, image_key="image", colorize_nlabels=None, monitor=None,
    ):
        super().__init__()
        ignore_keys = ignore_keys or []
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.uniform_sample_scale = uniform_sample_scale
        self.anisotropic = anisotropic
        self.p_prior = p_prior
        self.p_prior_s = p_prior_s
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
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

    def forward(self, input, scale=1, angle=0):
        posterior = self.encode(input)
        z = posterior.sample()
        if scale != 1:
            z = F.interpolate(z, scale_factor=scale, mode="bilinear", align_corners=False)
        if angle != 0:
            z = torch.rot90(z, k=angle, dims=[-1, -2])
        return self.decode(z), posterior, z

    def get_input(self, batch, key):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        return x.permute(0, 3, 1, 2).to(
            memory_format=torch.contiguous_format
        ).float()

    def _training_forward(self, batch):
        inputs = self.get_input(batch, self.image_key)
        if random.random() < self.p_prior:
            mode = "latent"
            if self.anisotropic:
                scale = tuple(random.choice([s / 32 for s in range(8, 32)]) for _ in range(2))
            else:
                scale = random.choice([s / 32 for s in range(8, 32)])
            angle = random.choice([1, 2, 3])
            reconstructions, posterior, _ = self(inputs, scale=scale, angle=angle)
            inputs = F.interpolate(
                inputs, scale_factor=scale, mode="bilinear", align_corners=False,
            )
            inputs = torch.rot90(inputs, k=angle, dims=[-1, -2])
        else:
            mode = "image"
            if random.random() < self.p_prior_s:
                scale = random.choice([s / 32 for s in range(8, 32)])
                inputs = F.interpolate(
                    inputs, scale_factor=scale, mode="bilinear", align_corners=False,
                )
            else:
                scale = 1
            reconstructions, posterior, _ = self(inputs)
        return inputs, reconstructions, posterior, scale, mode

    def compute_train_loss(self, batch, optimizer_idx, global_step):
        inputs, reconstructions, posterior, scale, mode = self._training_forward(batch)
        loss, log_dict = self.loss(
            inputs, reconstructions, posterior, optimizer_idx, global_step,
            last_layer=self.get_last_layer(), split="train",
        )
        metadata = {
            "scale": torch.as_tensor(scale, device=inputs.device).float().mean(),
            "latent": torch.as_tensor(float(mode == "latent"), device=inputs.device),
        }
        return loss, log_dict, metadata

    @torch.no_grad()
    def compute_validation_outputs(self, batch):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, _, _ = self(inputs)
        return {"inputs": inputs.detach(), "reconstructions": reconstructions.detach()}

    def build_optimizers(self, learning_rate, betas=(0.5, 0.9)):
        ae_parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        ae_parameters += list(self.quant_conv.parameters())
        ae_parameters += list(self.post_quant_conv.parameters())
        opt_ae = torch.optim.Adam(ae_parameters, lr=learning_rate, betas=betas)
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=learning_rate, betas=betas,
        )
        return opt_ae, opt_disc

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = {}
        x = self.get_input(batch, self.image_key).to(next(self.parameters()).device)
        if not only_inputs:
            if random.random() < 0.5:
                xrec, posterior, _ = self(x)
            else:
                xrec, posterior, _ = self(x, scale=0.5)
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

