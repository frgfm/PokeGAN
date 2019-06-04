import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from models import Discriminator, Generator
from preprocessing import scale, get_dataloader
from optim import get_labels, get_noise, get_discriminator_loss, get_generator_loss
from utils import print_samples, print_gradflow


def train_GAN(D, d_optimizer, G, g_optimizer, data_loader, fixed_z, n_epochs, train_on_gpu=True, loss_type='sgan',
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
            d_loss = get_discriminator_loss(D_real, D_fake, real_target, fake_target, loss_type=loss_type)

            # Backprop
            d_loss.backward()
            d_optimizer.step()

            ########################
            # GENERATOR TRAINING
            ########################

            g_optimizer.zero_grad()

            # Get discrimminator output for fake images
            z = get_noise((batch_size, fixed_z.size(1)), train_on_gpu)
            D_fake = D(G(z)).squeeze()

            # Compute loss
            g_loss = get_generator_loss(D_fake, real_target, D_real, loss_type=loss_type)

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


def train_ProGAN(min_scale, max_scale, lr_cycles,
                 batch_size, data_dir,
                 latent_feature_size, d_dict, g_dict,
                 train_on_gpu, g_lr_scaling_factor, betas, loss_type,
                 sample_size, logger, output_folder):

    nb_stages = int(np.log2(max_scale / min_scale)) + 1
    img_size = min_scale

    if len(lr_cycles) not in [1, nb_stages]:
        raise AssertionError("You need a cycle definition for each progressive stage or a single cycle for all stages")

    fixed_z = get_noise((sample_size, g_dict.get('z_size')))
    tot_epochs = 0

    for stage_idx in range(nb_stages):
        poke_loader = get_dataloader(batch_size, img_size, data_dir=data_dir)

        # Check number of sampling operations
        d_depth = int(np.log2(img_size / latent_feature_size))
        g_depth = d_depth + int(np.log2(d_dict.get('conv_dim') / g_dict.get('conv_dim')))
        # Recreate the nets
        d_channels = [3] + [2 ** idx * d_dict.get('conv_dim') for idx in range(d_depth)]
        D = Discriminator(d_channels, img_size, d_dict.get('conv_ksize'), d_dict.get('conv_stride'),
                          norm_layer=d_dict.get('norm_layer'), norm_fn=d_dict.get('norm_fn'), drop_rate=d_dict.get('drop_rate'),
                          weight_initializer=d_dict.get('weight_initializer'))

        g_channels = [2 ** (g_depth - 1 - idx) * g_dict.get('conv_dim') for idx in range(g_depth)] + [3]
        G = Generator(g_dict.get('z_size'), g_channels, img_size, g_dict.get('conv_ksize'), g_dict.get('conv_stride'),
                      norm_layer=g_dict.get('norm_layer'), norm_fn=g_dict.get('norm_fn'), drop_rate=g_dict.get('drop_rate'),
                      weight_initializer=g_dict.get('weight_initializer'))

        # Parameter loading & freezing
        if stage_idx > 0:
            # Load stage parameters & freeze previously trained layers
            if d_dict.get('norm_fn') is None:
                D.load_state_dict(d_state_dict, strict=False)
            else:  # state_loading for spectral norm
                tmp_dict = D.state_dict()
                tmp_dict.update({k: v for k, v in d_state_dict.items() if k in tmp_dict})
                D.load_state_dict(tmp_dict)
            D.freeze_layers(d_depth - 1)

            if g_dict.get('norm_fn') is None:
                G.load_state_dict(g_state_dict, strict=False)
            else:  # state_loading for spectral norm
                tmp_dict = G.state_dict()
                tmp_dict.update({k: v for k, v in d_state_dict.items() if k in tmp_dict})
                G.load_state_dict(tmp_dict)
            G.freeze_layers(g_depth - 1)

        # Move models to GPU
        if train_on_gpu:
            D.cuda()
            G.cuda()

        # Train the stage
        print(f"======================\nStage {img_size}x{img_size} ({stage_idx+1}/{nb_stages})\n======================")
        stage_cycle_idx = stage_idx if len(lr_cycles) > 1 else 0
        for cycle_idx, cycle_settings in enumerate(lr_cycles[stage_cycle_idx]):
            print(f"Cycle ({cycle_idx+1}/{len(lr_cycles[stage_cycle_idx])}) - {cycle_settings}")
            d_optimizer = optim.Adam(D.parameters(), cycle_settings.get('lr'), betas)
            g_optimizer = optim.Adam(G.parameters(), g_lr_scaling_factor * cycle_settings.get('lr'), betas)
            train_GAN(D, d_optimizer, G, g_optimizer, poke_loader, fixed_z,
                      cycle_settings.get('nb_epochs'), train_on_gpu,
                      loss_type, logger, log_every=10, output_folder=output_folder,
                      sample_print_freq=100, starting_epoch=tot_epochs)
            tot_epochs += cycle_settings.get('nb_epochs')
            # Model state saving (keep only sampling layers)
            d_state_dict = {p_name: p for p_name, p in D.state_dict().items()
                            if p_name.split('.')[0] == 'downblock'}
            g_state_dict = {p_name: p for p_name, p in G.state_dict().items()
                            if p_name.split('.')[0] == 'upblock'}

        img_size *= 2

    logger.close()

    return D, G
