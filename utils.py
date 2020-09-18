from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import math


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
