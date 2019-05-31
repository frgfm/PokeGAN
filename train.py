import torch
import numpy as np
from preprocessing import scale
from optim import get_labels, get_noise
from utils import print_samples, print_gradflow
import matplotlib.pyplot as plt


def train_GAN(D, d_optimizer, G, g_optimizer, data_loader, fixed_z, criterion, n_epochs, train_on_gpu=True,
                tb_logger=None, log_every=10, sample_print_freq=100, starting_epoch=0):
    '''Trains adversarial networks for some number of epochs
       param, D: the discriminator network
       param, G: the generator network
       param, n_epochs: number of epochs to train for
       param, print_every: when to print and record the models' losses
       return: D and G losses'''

    # keep track of loss and generated, "fake" samples

    # epoch training loop
    for epoch in range(n_epochs):

        # batch training loop
        for batch_i, (real_images, _) in enumerate(data_loader):

            batch_size = real_images.size(0)
            real_images = scale(real_images)
            img_size = real_images.size(-1)

            ########################
            # DISCRIMINATOR TRAINING
            ########################
            # 1. Train the discriminator on real and fake images
            d_optimizer.zero_grad()

            # Compute the discriminator losses on real images 
            if train_on_gpu:
                real_images = real_images.cuda()

            D_real = D(real_images)
            labels = get_labels(D_real.size(0), True, swap_prob=0.03, noise_norm=0.1)
            # move labels to GPU if available     
            if train_on_gpu:
                labels = labels.cuda()
            # calculate loss
            d_real_loss = criterion(D_real.squeeze(), labels)

            # 2. Train with fake images

            # Generate fake images
            z = get_noise((batch_size, fixed_z.size(1)))
            fake_images = G(z)

            # Compute the discriminator losses on fake images            
            D_fake = D(fake_images)
            labels = get_labels(D_fake.size(0), False, swap_prob=0.03, noise_norm=0.1)
            # move labels to GPU if available     
            if train_on_gpu:
                labels = labels.cuda()
            # calculate loss
            d_fake_loss = criterion(D_fake.squeeze(), labels)

            # add up loss and perform backprop
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            ########################
            # GENERATOR TRAINING
            ########################

            # 2. Train the generator with an adversarial loss
            g_optimizer.zero_grad()

            # Generate fake images
            z = get_noise((batch_size, fixed_z.size(1)))
            fake_images = G(z)

            # Compute the discriminator losses on fake images 
            # using flipped labels!
            D_fake = D(fake_images)
            labels = get_labels(D_fake.size(0), True, swap_prob=0.03, noise_norm=0.1)
            # move labels to GPU if available     
            if train_on_gpu:
                labels = labels.cuda()
            # calculate loss
            g_loss = criterion(D_fake.squeeze(), labels)

            # perform backprop
            g_loss.backward()
            g_optimizer.step()

        # Log stats
        info = dict(d_loss=d_loss.item(), g_loss=g_loss.item())
        current_iter = (starting_epoch + epoch) * len(data_loader) + batch_i
        if tb_logger is not None:
            tb_logger.add_scalars("losses", info, current_iter)
            # for name, param in D.named_parameters():
            #     if param.requires_grad and "bias" not in name:
            #         tb_logger.add_histogram(f"D.layer{name}", param.clone().cpu().data.numpy(), current_iter)
            # for name, param in G.named_parameters():
            #     if param.requires_grad and "bias" not in name:
            #         tb_logger.add_histogram(f"G.layer{name}", param.clone().cpu().data.numpy(), current_iter)


        # print discriminator and generator loss
        if (epoch + 1) % log_every == 0:
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                  epoch + 1, n_epochs, d_loss.item(), g_loss.item()))


        # Display samples
        if (epoch + 1) % sample_print_freq == 0:
            G.eval()  # for generating samples
            samples_z = G(fixed_z)
            G.train()  # back to training mode
            print_samples(samples_z, title=f"Epoch {epoch+1}", img_size=img_size)
            plt.show()
            # Images
            # if tb_logger:
                # x = torchvision.utils.make_grid(samples_z.clone().cpu(), normalize=True, scale_each=True)
                # tb_logger.add_image('gen_images', x, current_iter)
                # tb_logger.add_images('gen_images', samples_z.cpu().detach(), current_iter)
            plt.figure(figsize=(12, 5))
            plt.subplot(121)
            print_gradflow(D.named_parameters(), f"Gradient flow Discriminator")
            plt.subplot(122)
            print_gradflow(G.named_parameters(), f"Gradient flow Generator")
            plt.show()


def train_ProGAN():

    return 0