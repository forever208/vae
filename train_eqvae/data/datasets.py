"""Image datasets used by EQ-VAE training and validation."""

import os
from functools import partial

import albumentations
import cv2
import numpy as np
import PIL
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from data.degradation import degradation_fn_bsr, degradation_fn_bsr_light
except ModuleNotFoundError:
    from train_eqvae.data.degradation import (
        degradation_fn_bsr,
        degradation_fn_bsr_light,
    )


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_image_paths(root_dir, recursive=False):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {root_dir}")
    if recursive:
        paths = [
            os.path.relpath(os.path.join(dirpath, filename), root_dir)
            for dirpath, _, filenames in os.walk(root_dir)
            for filename in filenames
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
        ]
    else:
        paths = [
            filename
            for filename in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, filename))
            and os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
        ]
    paths.sort()
    if not paths:
        raise RuntimeError(f"No images found in dataset directory: {root_dir}")
    return paths


class _BaseImageDataset(Dataset):
    def _initialize(
        self, root_dir, dataset_name, size, degradation, downscale_f,
        min_crop_f, max_crop_f, random_crop,
    ):
        if dataset_name == "openimages":
            rel_paths = _list_image_paths(root_dir)
        elif dataset_name == "imagenet":
            rel_paths = _list_image_paths(root_dir, recursive=True)
        else:
            raise NotImplementedError(f"Unsupported dataset: {dataset_name}")
        self.labels = {
            "relative_file_path_": rel_paths,
            "file_path_": [os.path.join(root_dir, path) for path in rel_paths],
        }
        self.length = len(rel_paths)
        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert max_crop_f <= 1.0
        self.center_crop = not random_crop
        self.image_rescaler = albumentations.SmallestMaxSize(
            max_size=size, interpolation=cv2.INTER_AREA,
        )
        self.pil_interpolation = False
        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)
        elif degradation == "bsrgan_light":
            self.degradation_process = partial(
                degradation_fn_bsr_light, sf=downscale_f,
            )
        else:
            interpolation_fn = {
                "cv_nearest": cv2.INTER_NEAREST,
                "cv_bilinear": cv2.INTER_LINEAR,
                "cv_bicubic": cv2.INTER_CUBIC,
                "cv_area": cv2.INTER_AREA,
                "cv_lanczos": cv2.INTER_LANCZOS4,
                "pil_nearest": PIL.Image.NEAREST,
                "pil_bilinear": PIL.Image.BILINEAR,
                "pil_bicubic": PIL.Image.BICUBIC,
                "pil_box": PIL.Image.BOX,
                "pil_hamming": PIL.Image.HAMMING,
                "pil_lanczos": PIL.Image.LANCZOS,
            }[degradation]
            self.pil_interpolation = degradation.startswith("pil_")
            if self.pil_interpolation:
                self.degradation_process = partial(
                    TF.resize, size=self.LR_size, interpolation=interpolation_fn,
                )
            else:
                self.degradation_process = albumentations.SmallestMaxSize(
                    max_size=self.LR_size, interpolation=interpolation_fn,
                )

    def __len__(self):
        return self.length

    def _load_and_crop(self, index):
        example = {key: self.labels[key][index] for key in self.labels}
        image = Image.open(example["file_path_"])
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        min_side_len = min(image.shape[:2])
        crop_side_len = int(
            min_side_len
            * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        )
        crop = (
            albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)
            if self.center_crop
            else albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)
        )
        image = crop(image=image)["image"]
        image = self.image_rescaler(image=image)["image"]
        if self.pil_interpolation:
            low_resolution = np.array(
                self.degradation_process(PIL.Image.fromarray(image))
            ).astype(np.uint8)
        else:
            low_resolution = self.degradation_process(image=image)["image"]
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["LR_image"] = (low_resolution / 127.5 - 1.0).astype(np.float32)
        return example


class DatasetTrain(_BaseImageDataset):
    def __init__(
        self, train_dir="/data/openimages/target_dir/train/", size=None,
        dataset_name="openimages", degradation=None, downscale_f=4,
        min_crop_f=0.5, max_crop_f=1.0, random_crop=True,
    ):
        self._initialize(
            train_dir, dataset_name, size, degradation, downscale_f,
            min_crop_f, max_crop_f, random_crop,
        )

    def __getitem__(self, index):
        return self._load_and_crop(index)


class DatasetVal(_BaseImageDataset):
    def __init__(
        self, val_dir="/data/openimages/target_dir/validation/",
        dataset_name="openimages", size=None, degradation=None, downscale_f=4,
        min_crop_f=0.5, max_crop_f=1.0, random_crop=True,
        eval_preprocessing=False,
    ):
        self._initialize(
            val_dir, dataset_name, size, degradation, downscale_f,
            min_crop_f, max_crop_f, random_crop,
        )
        self.eval_preprocessing = eval_preprocessing
        self.eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

    def __getitem__(self, index):
        if not self.eval_preprocessing:
            return self._load_and_crop(index)
        example = {key: self.labels[key][index] for key in self.labels}
        image = Image.open(example["file_path_"])
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.eval_transform(image)
        low_resolution = TF.resize(
            image,
            [self.LR_size, self.LR_size],
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        example["image"] = image.permute(1, 2, 0).numpy().astype(np.float32)
        example["LR_image"] = (
            low_resolution.permute(1, 2, 0).numpy().astype(np.float32)
        )
        return example

