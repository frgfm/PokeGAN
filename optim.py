import torch
from torch.nn import BCEWithLogitsLoss
import numpy as np

IMPLEMENTED_LOSSES = ['sgan', 'rgan', 'ragan']


def get_labels(batch_size, real_label=True, swap_prob=0.03, noise_norm=0.1, train_on_gpu=True):

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

    if train_on_gpu:
        labels = labels.cuda()

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


def get_discriminator_loss(real_pred, fake_pred, real_target,
                           fake_target=None, loss_type='sgan'):

    if loss_type not in IMPLEMENTED_LOSSES:
        raise NotImplementedError(f"Loss type should be in {IMPLEMENTED_LOSSES}")

    if loss_type == 'sgan':
        adversarial_loss = BCEWithLogitsLoss()
        loss = 0.5 * (adversarial_loss(real_pred, real_target) +
                      adversarial_loss(fake_pred, fake_target))
    elif loss_type == 'rgan':
        adversarial_loss = BCEWithLogitsLoss()
        loss = 0.5 * (adversarial_loss(real_pred - fake_pred, real_target) +
                      adversarial_loss(fake_pred - real_pred, fake_target))
    elif loss_type == 'ragan':
        adversarial_loss = BCEWithLogitsLoss()
        loss = 0.5 * (adversarial_loss(real_pred - fake_pred.mean(), real_target) +
                      adversarial_loss(fake_pred - real_pred.mean(), fake_target))

    return loss


def get_generator_loss(fake_pred, criterion, real_target, real_pred=None, loss_type='sgan'):

    if loss_type not in IMPLEMENTED_LOSSES:
        raise NotImplementedError(f"Loss type should be in {IMPLEMENTED_LOSSES}")

    if loss_type == 'sgan':
        adversarial_loss = BCEWithLogitsLoss()
        loss = adversarial_loss(fake_pred, real_target)
    elif loss_type == 'rgan':
        adversarial_loss = BCEWithLogitsLoss()
        loss = adversarial_loss(fake_pred - real_pred, real_target)
    elif loss_type == 'ragan':
        adversarial_loss = BCEWithLogitsLoss()
        loss = adversarial_loss(fake_pred - real_pred.mean(), real_target)

    return loss
