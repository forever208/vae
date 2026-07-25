"""Adversarial loss helpers adapted from taming-transformers."""

import torch
import torch.nn.functional as F


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    return value if global_step < threshold else weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real, logits_fake):
    return 0.5 * (
        torch.mean(F.softplus(-logits_real))
        + torch.mean(F.softplus(logits_fake))
    )
