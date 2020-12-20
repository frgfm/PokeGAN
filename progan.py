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

    d_conv_dim, g_conv_dim = 32, 32

    # Stage
    num_stages = int(np.log2(args.size / args.min_size)) + 1
    img_sizes = [args.min_size * 2 ** idx for idx in range(num_stages)]
    d_state_dicts, g_state_dicts = None, None
    fixed_z = torch.randn((16, args.z_size)).cuda()

    for stage_idx, img_size in enumerate(img_sizes):

        print(f"======================\nStage {img_size}x{img_size} ({stage_idx+1}/{num_stages})\n======================")

        # Define transforms
        transform = Compose([
            Resize((img_size, img_size)),
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
        d_depth = int(np.log2(img_size / args.latent_size))
        d_chans = [3] + [d_conv_dim * 2 ** idx for idx in range(d_depth)]
        g_chans = [g_conv_dim * 2 ** idx for idx in range(d_depth)][::-1] + [3]
        # Recreate the nets
        discriminator = Discriminator(img_size, d_chans, 3, dropout=args.dropout)
        generator = Generator(args.z_size, img_size, g_chans, 5, dropout=args.dropout)

        # Load & freeze
        if d_state_dicts is not None:
            for _idx, state_dict in enumerate(d_state_dicts):
                discriminator[0][_idx].load_state_dict(state_dict)
            for p in discriminator[0][:-1].parameters():
                p.requires_grad_(False)
        if g_state_dicts is not None:
            # import ipdb; ipdb.set_trace()
            for _idx, state_dict in enumerate(g_state_dicts):
                generator[-1][_idx + 1].load_state_dict(state_dict)
            for p in generator[-1][1:].parameters():
                p.requires_grad_(False)

        discriminator, generator = discriminator.cuda(), generator.cuda()

        # Train our GAN
        gan_trainer = GANTrainer(discriminator, generator, img_size, args.z_size, train_loader)
        gan_trainer.fit_n_epochs(args.epochs, args.lr, args.weight_decay, args.label_smoothing, args.noise, args.swap, False)
        # Save state dicts
        d_state_dicts = [block.state_dict() for block in gan_trainer.discriminator[0]]
        g_state_dicts = [block.state_dict() for block in gan_trainer.generator[-1]]
        # Display samples
        gan_trainer.display_samples(fixed_z)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Pokemon ProGAN Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_path', type=str, help='path to dataset folder')
    parser.add_argument('--size', default=64, type=int, help='Image size to produce')
    parser.add_argument('--min-size', default=16, type=int, help='Image size at first stage')
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
