"""LPIPS perceptual loss adapted from taming-transformers."""

import hashlib
import os
from collections import namedtuple

import requests
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm


_URLS = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1",
}
_CHECKPOINTS = {"vgg_lpips": "vgg.pth"}
_MD5_DIGESTS = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}


def _download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as progress:
            with open(local_path, "wb") as output:
                for data in response.iter_content(chunk_size=chunk_size):
                    if data:
                        output.write(data)
                        progress.update(len(data))


def _md5_hash(path):
    with open(path, "rb") as source:
        return hashlib.md5(source.read()).hexdigest()


def _get_checkpoint_path(name, root, check=False):
    assert name in _URLS
    path = os.path.join(root, _CHECKPOINTS[name])
    if not os.path.exists(path) or (
        check and _md5_hash(path) != _MD5_DIGESTS[name]
    ):
        print(f"Downloading {name} model from {_URLS[name]} to {path}")
        _download(_URLS[name], path)
        digest = _md5_hash(path)
        assert digest == _MD5_DIGESTS[name], digest
    return path


class LPIPS(nn.Module):
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        checkpoint = _get_checkpoint_path(name, "pretrained_models")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.load_state_dict(state, strict=False)
        print(f"loaded pretrained LPIPS loss from {checkpoint}")

    def forward(self, input, target):
        input, target = self.scaling_layer(input), self.scaling_layer(target)
        outputs_input, outputs_target = self.net(input), self.net(target)
        linears = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        result = None
        for index, linear in enumerate(linears):
            feature_input = normalize_tensor(outputs_input[index])
            feature_target = normalize_tensor(outputs_target[index])
            value = spatial_average(
                linear.model((feature_input - feature_target) ** 2),
                keepdim=True,
            )
            result = value if result is None else result + value
        return result


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "shift", torch.tensor([-.030, -.088, -.188])[None, :, None, None],
        )
        self.register_buffer(
            "scale", torch.tensor([.458, .448, .450])[None, :, None, None],
        )

    def forward(self, input):
        return (input - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, 1, 0, bias=False)]
        self.model = nn.Sequential(*layers)


class vgg16(nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        features = models.vgg16(pretrained=pretrained).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for index in range(4):
            self.slice1.add_module(str(index), features[index])
        for index in range(4, 9):
            self.slice2.add_module(str(index), features[index])
        for index in range(9, 16):
            self.slice3.add_module(str(index), features[index])
        for index in range(16, 23):
            self.slice4.add_module(str(index), features[index])
        for index in range(23, 30):
            self.slice5.add_module(str(index), features[index])
        self.N_slices = 5
        if not requires_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, input):
        relu1_2 = self.slice1(input)
        relu2_2 = self.slice2(relu1_2)
        relu3_3 = self.slice3(relu2_2)
        relu4_3 = self.slice4(relu3_3)
        relu5_3 = self.slice5(relu4_3)
        outputs = namedtuple(
            "VggOutputs",
            ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"],
        )
        return outputs(relu1_2, relu2_2, relu3_3, relu4_3, relu5_3)


def normalize_tensor(value, eps=1e-10):
    return value / (torch.sqrt(torch.sum(value ** 2, dim=1, keepdim=True)) + eps)


def spatial_average(value, keepdim=True):
    return value.mean([2, 3], keepdim=keepdim)
