import unittest
import torch
from torchgan import models


class ModelsTester(unittest.TestCase):
    def test_discriminator(self):

        num_batches = 8
        img_size = 16

        discriminator = models.Discriminator(img_size, [3, 32, 64], 3, dropout=0.3)

        # Check integrity of results
        self.assertIsInstance(discriminator, torch.nn.Sequential)

        # Check output
        mock_x = torch.rand((num_batches, 3, img_size, img_size))
        discriminator.eval()
        with torch.no_grad():
            out = discriminator(mock_x)
            self.assertEqual(out.shape, (num_batches, 1))

        # Check backprop
        self.assertIsNone(discriminator[-1].weight.grad)
        discriminator.train()
        out = discriminator(mock_x)
        out.sum().backward()
        self.assertIsInstance(discriminator[-1].weight.grad, torch.Tensor)

    def test_generator(self):
        num_batches = 8
        img_size = 16
        z_size = 96

        generator = models.Generator(z_size, img_size, [64, 32, 3], 5, dropout=0.3)

        # Check integrity of results
        self.assertIsInstance(generator, torch.nn.Sequential)

        # Check output
        mock_x = torch.rand((num_batches, z_size))
        generator.eval()
        with torch.no_grad():
            out = generator(mock_x)
            self.assertEqual(out.shape, (num_batches, 3, img_size, img_size))

        # Check backprop
        self.assertIsNone(generator[0].weight.grad)
        generator.train()
        out = generator(mock_x)
        out.sum().backward()
        self.assertIsInstance(generator[0].weight.grad, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
