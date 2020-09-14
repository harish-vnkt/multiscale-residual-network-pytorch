from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


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
