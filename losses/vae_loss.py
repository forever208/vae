"""VAE reconstruction, perceptual, KL and adversarial loss."""

import torch
import torch.nn as nn

from losses.gan import adopt_weight, hinge_d_loss, vanilla_d_loss
from losses.lpips import LPIPS
from models.discriminator import NLayerDiscriminator, weights_init


class LPIPSWithDiscriminator(nn.Module):
    def __init__(
        self, disc_start, logvar_init=0.0, kl_weight=1.0,
        pixelloss_weight=1.0, disc_num_layers=3, disc_in_channels=3,
        disc_factor=1.0, disc_weight=1.0, perceptual_weight=1.0,
        use_actnorm=False, disc_conditional=False, disc_loss="hinge",
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        layer = last_layer if last_layer is not None else self.last_layer[0]
        nll_grads = torch.autograd.grad(nll_loss, layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, layer, retain_graph=True)[0]
        weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        return torch.clamp(weight, 0.0, 1e4).detach() * self.discriminator_weight

    def forward(
        self, inputs, reconstructions, posteriors, optimizer_idx, global_step,
        last_layer=None, cond=None, split="train", weights=None,
    ):
        pixel_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        rec_loss = pixel_loss
        p_loss = torch.zeros((), device=inputs.device)
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous(),
            )
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = weights * nll_loss if weights is not None else nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        pixel_contribution = pixel_loss / torch.exp(self.logvar)
        lpips_contribution = (
            self.perceptual_weight * p_loss * torch.ones_like(pixel_loss)
            / torch.exp(self.logvar)
        )
        logvar_contribution = self.logvar * torch.ones_like(pixel_loss)
        if weights is not None:
            pixel_contribution *= weights
            lpips_contribution *= weights
            logvar_contribution *= weights
        weighted_pixel_loss = torch.sum(pixel_contribution) / pixel_contribution.shape[0]
        weighted_lpips_loss = torch.sum(lpips_contribution) / lpips_contribution.shape[0]
        weighted_logvar_loss = torch.sum(logvar_contribution) / logvar_contribution.shape[0]
        kl_values = posteriors.kl()
        kl_loss = torch.sum(kl_values) / kl_values.shape[0]

        if optimizer_idx == 0:
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1)
                )
            g_loss = -torch.mean(logits_fake)
            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer,
                    )
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0, device=inputs.device)
            else:
                d_weight = torch.tensor(0.0, device=inputs.device)
            disc_factor = adopt_weight(
                self.disc_factor, global_step,
                threshold=self.discriminator_iter_start,
            )
            weighted_kl_loss = self.kl_weight * kl_loss
            weighted_g_loss = d_weight * disc_factor * g_loss
            loss = weighted_nll_loss + weighted_kl_loss + weighted_g_loss
            log = {
                f"{split}/total_loss": loss.detach().mean(),
                f"{split}/logvar": self.logvar.detach(),
                f"{split}/kl_loss": kl_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/pixel_loss": pixel_loss.detach().mean(),
                f"{split}/lpips_loss": p_loss.detach().mean(),
                f"{split}/weighted_pixel_loss": weighted_pixel_loss.detach().mean(),
                f"{split}/weighted_lpips_loss": weighted_lpips_loss.detach().mean(),
                f"{split}/weighted_logvar_loss": weighted_logvar_loss.detach().mean(),
                f"{split}/weighted_kl_loss": weighted_kl_loss.detach().mean(),
                f"{split}/weighted_g_loss": weighted_g_loss.detach().mean(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/disc_factor": torch.tensor(disc_factor),
                f"{split}/g_loss": g_loss.detach().mean(),
            }
            return loss, log

        if optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous().detach(), cond), dim=1)
                )
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1)
                )
            disc_factor = adopt_weight(
                self.disc_factor, global_step,
                threshold=self.discriminator_iter_start,
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            return d_loss, {
                f"{split}/disc_loss": d_loss.detach().mean(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean(),
            }

