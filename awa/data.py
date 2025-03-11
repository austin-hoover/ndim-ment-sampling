import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import downscale_local_mean

from phase_space_reconstruction.modeling import ImageDataset3D
from phase_space_reconstruction.utils import split_2screen_dset


def collect_images_flat(data: ImageDataset3D) -> torch.Tensor:
    images = torch.clone(data.images)
    images = torch.sum(images, axis=3)  # average over multi-shot images
    size = images.shape[0] * images.shape[1] * images.shape[2]
    images = images.reshape(size, images.shape[-2], images.shape[-1])
    return images


def collect_params_flat(data: ImageDataset3D) -> torch.Tensor:
    params = torch.clone(data.params)
    size = params.shape[0] * params.shape[1] * params.shape[2]
    params = params.reshape(size, params.shape[-1])
    return params


def threshold_images(images: torch.Tensor, thresh: float = 0.0) -> torch.Tensor:
    if thresh <= 0.0:
        return images
        
    images_new = []
    for image in images.numpy():
        image_new = np.copy(image)
        image_new[image_new < thresh] = 0.0
        images_new.append(image_new)
    images_new = np.array(images_new)
    images_new = torch.from_numpy(images_new)
    return images_new


def downscale_images(images: torch.Tensor, factor: int = 1) -> torch.Tensor:
    if factor <= 1:
        return images

    images_new = []
    for image in images.numpy():
        image_new = np.copy(image)
        image_new = downscale_local_mean(image, (factor, factor))
        images_new.append(image_new)
    images_new = np.array(images_new)
    images_new = torch.from_numpy(images_new)
    return images_new


def load_data_flat(filename: str, downscale: int = 1, thresh: float = 0) -> dict:
    # Load image datasets
    data = torch.load(filename)
    train_data, test_data = split_2screen_dset(data)

    # Collect images and parameters for 3D scan (quad, tdc, dipole)
    train_images = collect_images_flat(train_data)
    train_params = collect_params_flat(train_data)
    test_images = collect_images_flat(test_data)
    test_params = collect_params_flat(test_data)

    # Threshold images
    train_images = threshold_images(train_images, thresh)
    test_images = threshold_images(test_images, thresh)
    
    # Downscale images
    train_images = downscale_images(train_images, downscale)
    test_images = downscale_images(test_images, downscale)

    output = {
        "train": {
            "images": train_images,
            "params": train_params,
        },
        "test": {
            "images": test_images,
            "params": test_params,
        },
    }
    return output
