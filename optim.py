import torch
import numpy as np


def get_labels(batch_size, real_label=True, swap_prob=0.03, noise_norm=0.1):

    # Label smoothing
    if real_label:
        labels = np.ones(batch_size) + np.random.uniform(-noise_norm, noise_norm)
    else:
        labels = np.zeros(batch_size) + np.random.uniform(0, 2 * noise_norm)

    # Label swaping
    distrib = np.random.uniform(0, 1, batch_size)
    swap_idxs = np.argwhere(distrib < swap_prob).flatten()
    labels[swap_idxs] = float(not real_label)
    labels = torch.from_numpy(labels).float()

    return labels


def get_noise(output_shape, train_on_gpu=True):

    # z = rand_spherical(batch_size, fixed_z.size(1), norm=1)
    z = np.random.normal(0, 1, size=output_shape)
    z = torch.from_numpy(z).float()
    # move x to GPU, if available
    if train_on_gpu:
        z = z.cuda()
    return z


def rand_spherical(b_size, v_size, norm=1):
    z = np.random.uniform(-norm, norm, size=(b_size, v_size))
    z /= np.linalg.norm(z, axis=1)[:, None]
    return z


def normal_initialization(m, mean=0, std=0.2):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution 
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network    
    """
    # classname will be something like:
    # `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__

    # Apply initial weights to convolutional and linear layers
    if (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        torch.nn.init.normal_(m.weight.data, mean, std)
