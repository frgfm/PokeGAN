import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import (Compose, Resize, ColorJitter, RandomRotation, RandomHorizontalFlip, ToTensor,
                                    Normalize)

from torchgan.models import Discriminator, Generator
from torchgan.gan import GANTrainer
from torchgan.utils import print_samples


def main(args):

    # Set device
    if isinstance(args.device, int):
        if not torch.cuda.is_available():
            raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
        if args.device >= torch.cuda.device_count():
            raise ValueError("Invalid device index")
        torch.cuda.set_device(args.device)

    # Define transforms
    transform = Compose([
        Resize((args.size, args.size)),
        ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1),
        RandomRotation(15),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # TODO: batch transforms Jitter, Rotation, Flip & Normalize

    # define datasets using ImageFolder
    train_set = ImageFolder(args.data_path, transform)

    # create and return DataLoaders
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, drop_last=True,
                              sampler=RandomSampler(train_set), num_workers=args.workers, pin_memory=True)

    # Dynamically define models from image size and latent space size
    d_conv_dim, g_conv_dim = 32, 32
    d_depth = int(np.log2(args.size / args.latent_size))
    d_chans = [3] + [d_conv_dim * 2 ** idx for idx in range(d_depth + 1)]
    g_chans = [g_conv_dim * 2 ** idx for idx in range(d_depth + 1)][::-1] + [3]
    # Recreate the nets
    D = Discriminator(args.size, d_chans, 3, dropout=args.dropout).cuda()
    G = Generator(args.z_size, args.size, g_chans, 5, dropout=args.dropout).cuda()
    print(D)
    print(G)

    # Train our GAN
    gan_trainer = GANTrainer(D, G, args.z_size, train_loader)
    gan_trainer.fit_n_epochs(args.epochs, args.lr, args.weight_decay, args.label_smoothing, args.noise, args.swap)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Pokemon GAN Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_path', type=str, help='path to dataset folder')
    parser.add_argument('--size', default=64, type=int, help='Image size to produce')
    parser.add_argument('--device', default=0, type=int, help='device')

    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')
    parser.add_argument('--z-size', default=96, type=int, help='number of features fed to the generator')
    parser.add_argument('--latent-size', default=4, type=int, help='latent feature map size')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float, help='weight decay', dest='weight_decay')
    parser.add_argument('--ls', '--label-smoothing', default=0.1, type=float, help='label smoothing',
                        dest='label_smoothing')
    parser.add_argument('--noise', default=0.1, type=float, help='Norm of the noise added to labels')
    parser.add_argument('--swap', default=0.03, type=float, help='Probability of swapping labels')

    parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=400, type=int, help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
