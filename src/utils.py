from pathlib import Path
from itertools import chain
from typing import List
import numpy as np
import pandas as pd
import random
import pickle
import torch
import torch.nn as nn
import math


def get_project_root(): 
    """
    Returns project root directory from this script nested in the commons folder.
    """
    return Path(__file__).parent.parent


def load_images(dataset:str='cifar10', batch_size:int=32, with_labels:bool=False, verbose:int=None)->object:
    """
    Loads images from a specified dataset and batch size.

    Args:
        dataset (str): The name of the dataset to load images from. Can be one of "cifar10", "cifar100", or "imagenet".
        batch_size (int): The batch size of the images to load. Can be either 32 or 64.
        with_labels (bool): Whether or not to return labelled examples. Defaults to False.
        verbose (int): Whether or not to print a message when a batch is loaded. Defaults to None.

    Returns:
        object: If with_labels is True, returns labelled examples. Otherwise, returns data with no labels.
    """
    if dataset not in ["cifar10", "cifar100", "imagenet"]:
        if 'imagenet' not in dataset.lower():
            raise ValueError('Please specify a valid dataset. Should be one of cifar10, cifar100, ImageNet')
        else:
            dataset = 'imagenet'
    if batch_size not in [32, 64]:
            raise ValueError(f"Batch size: {batch_size} not accepted. Can only be 32 or 64.")
    # sampling one random batch randomly
    random_batch = random.randrange(10)

    with open(str(get_project_root()) + "/archive/" + f'data/{dataset}__batch{batch_size}_{random_batch}', 'rb') as pickle_file:
        images = pickle.load(pickle_file)
        if verbose: 
            print(f'Batch #{random_batch} loaded.')

    # returning one of the random batches generated randomly
    if with_labels:
        return images  # returns labelled examples
    else: 
        return images[0].float()  # only returns data, with no labels. Imagenet tensors are in uint8 hence mapping to floats


def correlation(tensor:torch.tensor)->float:
    """Compute correlation coefficient on a tensor, based on
    https://math.stackexchange.com/a/1393907

    Args:
        tensor (torch.tensor):

    Returns:
        float: Pearson correlation coefficient
    """
    tensor = tensor.double()
    r1 = torch.tensor(range(1, tensor.shape[0] + 1)).double()
    r2 = torch.tensor([i*i for i in range(1, tensor.shape[0] + 1)]).double()
    j = torch.ones(tensor.shape[0]).double()
    n = torch.matmul(torch.matmul(j, tensor), j.T).double()
    x = torch.matmul(torch.matmul(r1, tensor), j.T)
    y = torch.matmul(torch.matmul(j, tensor), r1.T)
    x2 = torch.matmul(torch.matmul(r2, tensor), j.T)
    y2 = torch.matmul(torch.matmul(j, tensor), r2.T)
    xy = torch.matmul(torch.matmul(r1, tensor), r1.T)
    
    corr = (n * xy - x * y) / (torch.sqrt(n * x2 - x**2) * torch.sqrt(n * y2 - y**2))

    return corr.item()


def kaiming_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def init_model(model):
    """Applies kaiming normal weights initialization to input model."""
    model.apply(kaiming_normal)
    return model


def seed_all(seed:int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_scientific_notation(number: float) -> str:
    """
    Converts number to scientific notation with one digit.
    For instance, 5000 becomes '5e3' and 123.45 becomes '1.2e2'
    """
    exponent = math.floor(math.log10(abs(number)))
    mantissa = round(number / (10 ** exponent), 1)

    # Format as string
    return f"{mantissa}e{exponent}"

