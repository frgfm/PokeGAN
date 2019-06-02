import os
import torch
import numpy as np
from preprocessing import scale
import matplotlib.pyplot as plt
import torchvision

from optim import get_labels, get_noise, get_discriminator_loss, get_generator_loss
from utils import print_samples, print_gradflow


def train_GAN(D, d_optimizer, G, g_optimizer, data_loader, fixed_z, criterion, n_epochs, train_on_gpu=True,
              tb_logger=None, log_every=10, output_folder='training', sample_print_freq=100, starting_epoch=0):
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
            # Move images to GPU
            if train_on_gpu:
                real_images = real_images.cuda()
            img_size = real_images.size(-1)

            # Targets (smooth & swap labels)
            real_target = get_labels(batch_size, True, swap_prob=0.03, noise_norm=0.1, train_on_gpu=train_on_gpu)
            fake_target = get_labels(batch_size, False, swap_prob=0.03, noise_norm=0.1, train_on_gpu=train_on_gpu)

            ########################
            # DISCRIMINATOR TRAINING
            ########################

            d_optimizer.zero_grad()

            # Get discrimminator output for real and fake images
            D_real = D(real_images).squeeze()
            # Generate fake images
            z = get_noise((batch_size, fixed_z.size(1)), train_on_gpu)
            D_fake = D(G(z)).squeeze()

            # Compute loss
            d_loss = get_discriminator_loss(D_real, D_fake, criterion, real_target, fake_target, loss_type=loss_type)

            # Backprop
            d_loss.backward()
            d_optimizer.step()

            ########################
            # GENERATOR TRAINING
            ########################

            g_optimizer.zero_grad()

            # Resample target
            real_target = get_labels(batch_size, True, swap_prob=0.03, noise_norm=0.1, train_on_gpu=train_on_gpu)

            # Get discrimminator output for fake images
            z = get_noise((batch_size, fixed_z.size(1)), train_on_gpu)
            D_fake = D(G(z)).squeeze()

            # Compute loss
            g_loss = get_generator_loss(D_fake, criterion, real_target, D_real, loss_type=loss_type)

            # Backprop
            g_loss.backward()
            g_optimizer.step()

        # Tensorboard monitoring
        current_iter = (starting_epoch + epoch) * len(data_loader) + batch_i
        if tb_logger is not None:
            # Network losses
            tb_logger.add_scalars(f"{loss_type}_loss",
                                  dict(d_loss=d_loss.item(), g_loss=g_loss.item()),
                                  current_iter)
            # Histograms of parameters value and gradients
            for name, param in D.named_parameters():
                if param.requires_grad and "bias" not in name:
                    tag = f"D/{name.replace('.', '/')}"
                    tb_logger.add_histogram(f"{tag}/value", param.cpu(), current_iter)
                    tb_logger.add_histogram(f"{tag}/grad", param.grad.cpu(), current_iter)
            for name, param in G.named_parameters():
                if param.requires_grad and "bias" not in name:
                    tag = f"G/{name.replace('.', '/')}"
                    tb_logger.add_histogram(f"{tag}/value", param.cpu(), current_iter)
                    tb_logger.add_histogram(f"{tag}/grad", param.grad.cpu(), current_iter)

        # Console printing
        if (epoch + 1) % log_every == 0:
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                  epoch + 1, n_epochs, d_loss.item(), g_loss.item()))

        # Sample displaying
        if (epoch + 1) % sample_print_freq == 0:
            # Generate samples
            G.eval()
            samples_z = G(fixed_z)
            G.train()
            # Console display
            print_samples(samples_z, img_size=img_size)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'samples', f"stage{img_size}_epoch{starting_epoch + epoch + 1}.png"),
                        transparent=True)
            plt.show()
            # Tensorboard display
            if tb_logger is not None:
                x = torchvision.utils.make_grid(samples_z.cpu(), normalize=True, scale_each=True)
                tb_logger.add_image(f"samples", x, current_iter)
            # Gradient flow
            plt.figure(figsize=(12, 7))
            plt.subplot(121)
            print_gradflow(D.named_parameters(), "Discriminator gradient flow")
            plt.subplot(122)
            print_gradflow(G.named_parameters(), "Generator gradient flow")
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'gradflow', f"stage{img_size}_epoch{starting_epoch + epoch + 1}.png"),
                        transparent=True)
            plt.show()

            # Save model states
            torch.save(D.state_dict(), os.path.join(output_folder, 'model_states', f"D_stage{img_size}_epoch{starting_epoch + epoch + 1}.pth"))
            torch.save(G.state_dict(), os.path.join(output_folder, 'model_states', f"G_stage{img_size}_epoch{starting_epoch + epoch + 1}.pth"))


def train_ProGAN():

    return 0