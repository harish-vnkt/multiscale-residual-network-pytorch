from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import math
import torch
import torch.nn as nn
import os
import cv2


def get_samplers(dataset, seed=42):

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = 100

    np.random.seed(seed)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler


def get_psnr(hr_ground_truth, hr_predicted):

    error = hr_ground_truth - hr_predicted
    squared_error = error ** 2
    mean_squared_error = squared_error.mean((-1, -2, -3), keepdim=False)
    psnr = 10 * math.log10((255 ** 2) / mean_squared_error.item())
    return psnr


class MeanShift(nn.Conv2d):

    def __init__(self, sign=-1, rgb_mean=(0.4040, 0.4371, 0.4488)):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)

        rgb_std = (1.0, 1.0, 1.0)

        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * 255 * torch.Tensor(rgb_mean) / std

        for p in self.parameters():
            p.requires_grad = False


def write_results(predictions, results_dir, image_name):

    image_path = os.path.join(results_dir, image_name)
    image_tensor = predictions[0, :, :, :]
    image_tensor.permute(1, 2, 0)
    image_tensor = image_tensor.cpu().numpy()

    cv2.imwrite(image_path, image_tensor)