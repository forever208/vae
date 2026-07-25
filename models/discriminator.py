"""PatchGAN discriminator adapted from taming-transformers."""

import functools

import torch
import torch.nn as nn


class ActNorm(nn.Module):
    def __init__(
        self, num_features, logdet=False, affine=True,
        allow_reverse_init=False,
    ):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flattened = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = flattened.mean(1)[None, :, None, None]
            std = flattened.std(1)[None, :, None, None]
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        squeeze = len(input.shape) == 2
        if squeeze:
            input = input[:, :, None, None]
        _, _, height, width = input.shape
        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
        output = self.scale * (input + self.loc)
        if squeeze:
            output = output.squeeze(-1).squeeze(-1)
        if self.logdet:
            logdet = height * width * torch.sum(torch.log(torch.abs(self.scale)))
            return output, logdet * torch.ones(input.shape[0]).to(input)
        return output

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is disabled by default."
                )
            self.initialize(output)
            self.initialized.fill_(1)
        squeeze = len(output.shape) == 2
        if squeeze:
            output = output[:, :, None, None]
        result = output / self.scale - self.loc
        return result.squeeze(-1).squeeze(-1) if squeeze else result


def weights_init(module):
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        super().__init__()
        norm_layer = ActNorm if use_actnorm else nn.BatchNorm2d
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        kernel_width, padding = 4, 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_width, 2, padding),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for layer in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** layer, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev, ndf * nf_mult,
                    kernel_width, 2, padding, bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev, ndf * nf_mult,
                kernel_width, 1, padding, bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_width, 1, padding),
        ]
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        return self.main(input)
