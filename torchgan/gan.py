import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, Optimizer
from typing import Tuple
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import ConsoleMasterBar

from .utils import print_samples


__all__ = ['GANTrainer']

IMPLEMENTED_LOSSES = ['sgan', 'lsgan', 'wgan', 'rgan', 'ragan']


class GANTrainer:
    def __init__(
        self,
        discriminator: nn.Module,
        generator: nn.Module,
        img_size: int,
        z_size: int,
        train_loader: DataLoader,
        loss_type: str = 'sgan'
    ) -> None:
        self.discriminator: nn.Module = discriminator
        self.generator: nn.Module = generator
        self.img_size: int = img_size
        self.z_size: int = z_size
        self.train_loader: DataLoader = train_loader
        self.loss_type = loss_type

    def fit_n_epochs(
        self,
        num_epochs: int,
        lr: float = 1e-3,
        wd: float = 0.,
        label_smoothing: float = 0.1,
        noise: float = 0.1,
        swap_prob: float = 0.03,
        display_samples: bool = True,
    ) -> None:
        d_optimizer = Adam([p for p in self.discriminator.parameters() if p.requires_grad], lr,
                           betas=(0.5, 0.999), eps=1e-6, weight_decay=wd)
        g_optimizer = Adam([p for p in self.generator.parameters() if p.requires_grad], lr,
                           betas=(0.5, 0.999), eps=1e-6, weight_decay=wd)

        # keep track of loss and generated, "fake" samples
        fixed_z = torch.randn((16, self.z_size)).cuda()

        mb = master_bar(range(num_epochs))
        for epoch in mb:
            d_loss, g_loss = self.fit_one_epoch(d_optimizer, g_optimizer, mb, label_smoothing, noise, swap_prob)
            mb.main_bar.comment = f"Epoch {epoch + 1}/{num_epochs}"
            mb.write(f"Epoch {epoch + 1}/{num_epochs} - D loss: {d_loss.item():.4} | G loss: {g_loss.item():.4}")

            # Generate samples
            if display_samples and ((epoch + 1) % 100 == 0):
                self.display_samples(fixed_z)

    @torch.no_grad()
    def display_samples(self, fixed_z: Tensor) -> None:
        self.generator.eval()
        samples_z = self.generator(fixed_z)
        # Console display
        print_samples(samples_z, img_size=self.img_size)

    def fit_one_epoch(
        self,
        d_optimizer: Optimizer,
        g_optimizer: Optimizer,
        mb: ConsoleMasterBar,
        label_smoothing: float = 0.1,
        noise: float = 0.1,
        swap_prob: float = 0.03
    ) -> Tuple[Tensor, Tensor]:

        self.discriminator.train()
        self.generator.train()
        pb = progress_bar(self.train_loader, parent=mb)
        for real_images, _ in pb:
            real_images = real_images.cuda()

            # Target
            real_target, fake_target = self._get_labels(real_images.size(0), label_smoothing, noise, swap_prob)
            real_target, fake_target = real_target.cuda(), fake_target.cuda()

            ########################
            # DISCRIMINATOR TRAINING
            ########################

            d_optimizer.zero_grad()

            # Get discrimminator output for real and fake images
            d_real = self.discriminator(real_images)
            # Generate fake images
            z = torch.randn((real_images.size(0), self.z_size)).cuda()
            d_fake = self.discriminator(self.generator(z))

            # Compute loss
            d_loss = discriminator_loss(d_real.squeeze(), d_fake.squeeze(), real_target, fake_target, self.loss_type)

            # Backprop
            d_loss.backward()
            d_optimizer.step()

            ########################
            # GENERATOR TRAINING
            ########################

            g_optimizer.zero_grad()

            # Get discrimminator output for fake images
            z = torch.randn((real_images.size(0), self.z_size)).cuda()
            d_fake = self.discriminator(self.generator(z))

            # Compute loss
            g_loss = generator_loss(d_fake.squeeze(), real_target, d_real.squeeze(), self.loss_type)

            # Backprop
            g_loss.backward()
            g_optimizer.step()

            # Console printing
            pb.comment = f"D loss: {d_loss.item():.4} | G loss: {g_loss.item():.4}"

        return d_loss, g_loss

    @staticmethod
    def _get_labels(
        batch_size: int,
        label_smoothing: float = 0.1,
        noise: float = 0.1,
        swap_prob: float = 0.03
    ) -> Tuple[Tensor, Tensor]:

        # Targets (smooth & swap labels)
        real_target = torch.ones(batch_size) - label_smoothing
        fake_target = torch.zeros(batch_size) + label_smoothing
        # Noise
        real_target += noise * (torch.rand(batch_size) - 0.5)
        fake_target += noise * (torch.rand(batch_size) - 0.5)
        # Label swaping
        is_swaped = torch.rand(batch_size) < swap_prob
        real_target[is_swaped] = 1 - real_target[is_swaped]
        is_swaped = torch.rand(batch_size) < swap_prob
        fake_target[is_swaped] = 1 - fake_target[is_swaped]

        return real_target, fake_target


def discriminator_loss(
    real_pred: Tensor,
    fake_pred: Tensor,
    real_target: Tensor,
    fake_target: Tensor,
    loss_type: str = 'sgan'
) -> Tensor:

    if loss_type not in IMPLEMENTED_LOSSES:
        raise NotImplementedError(f"Loss type should be in {IMPLEMENTED_LOSSES}")

    if loss_type == 'sgan':
        loss = 0.5 * (F.binary_cross_entropy_with_logits(real_pred, real_target) +
                      F.binary_cross_entropy_with_logits(fake_pred, fake_target))
    elif loss_type == 'lsgan':
        loss = 0.5 * (F.mse_loss(real_pred, real_target) +
                      F.mse_loss(fake_pred, fake_target))
    elif loss_type == 'wgan':
        loss = fake_pred.mean() - real_pred.mean()
    elif loss_type == 'rgan':
        loss = 0.5 * (F.binary_cross_entropy_with_logits(real_pred - fake_pred, real_target) +
                      F.binary_cross_entropy_with_logits(fake_pred - real_pred, fake_target))
    elif loss_type == 'ragan':
        loss = 0.5 * (F.binary_cross_entropy_with_logits(real_pred - fake_pred.mean(), real_target) +
                      F.binary_cross_entropy_with_logits(fake_pred - real_pred.mean(), fake_target))

    return loss


def generator_loss(
    fake_pred: Tensor,
    real_target: Tensor,
    real_pred: Tensor,
    loss_type: str = 'sgan'
) -> Tensor:

    if loss_type not in IMPLEMENTED_LOSSES:
        raise NotImplementedError(f"Loss type should be in {IMPLEMENTED_LOSSES}")

    if loss_type == 'sgan':
        loss = F.binary_cross_entropy_with_logits(fake_pred, real_target)
    elif loss_type == 'lsgan':
        loss = F.mse_loss(fake_pred, real_target)
    elif loss_type == 'wgan':
        loss = -fake_pred.mean()
    elif loss_type == 'rgan':
        loss = F.binary_cross_entropy_with_logits(fake_pred - real_pred, real_target)
    elif loss_type == 'ragan':
        loss = F.binary_cross_entropy_with_logits(fake_pred - real_pred.mean(), real_target)

    return loss
