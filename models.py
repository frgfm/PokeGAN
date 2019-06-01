import torch.nn as nn
from collections import OrderedDict


class Flatten(nn.Module):
    def __init__(self, *args):
        super(Flatten, self).__init__()
        self.output_shape = (-1,) + args

    def forward(self, x):
        # return x.view(-1, *self.output_shape)
        return x.view(*self.output_shape)


def downsampler(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                normalization_block=None, use_dropout=True):

    layers_dict = OrderedDict()
    layers_dict['conv'] = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, bias=False)
    if normalization_block is not None:
        layers_dict['norm'] = normalization_block(out_channels)
    layers_dict['activation'] = nn.LeakyReLU(0.2)
    if use_dropout:
        layers_dict['dropout'] = nn.Dropout(0.3)

    return nn.Sequential(layers_dict)


class Discriminator(nn.Module):

    def __init__(self, conv_channels, input_size=32, kernel_size=3, stride=2,
                 normalization_block=None, use_dropout=True, init_weights=None):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()

        # Determinate optimal operations parameters
        pad = (kernel_size - 1) // stride
        current_size = input_size

        # Stack all layers
        layers_dict = OrderedDict()
        for layer_idx, out_channels in enumerate(conv_channels[1:]):

            # Add layers
            layers_dict[f"layer_{layer_idx+1}"] = downsampler(conv_channels[layer_idx], out_channels, kernel_size,
                                                              stride=stride, padding=pad,
                                                              normalization_block=normalization_block if layer_idx > 0 else None,
                                                              use_dropout=use_dropout)
            # Update temp parameters
            current_size = current_size // stride
        self.downblock = nn.Sequential(layers_dict)
        self.flatten = Flatten(current_size ** 2 * out_channels)
        self.fc = nn.Linear(current_size ** 2 * out_channels, 1)

        # Init weights
        if init_weights is not None:
            self.apply(init_weights)

    def freeze_layers(self, nb_layers=None):

        for name, param in self.named_parameters():
            # focus on downsampling blocks
            if name.split('.')[0].split('_')[0] == 'downblock':
                # freeze all nb_layers first layers
                if int(name.split('.')[1].split('_')[-1]) < nb_layers:
                    param.requires_grad = False

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior
        x = self.downblock(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


def upsampler(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
              normalization_block=None, activation=None, use_dropout=True):

    layers_dict = OrderedDict()
    layers_dict['tconv'] = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                              stride=stride, padding=padding, bias=False, output_padding=1)
    if normalization_block is not None:
        layers_dict['norm'] = normalization_block(out_channels)
    if activation is not None:
        layers_dict['activation'] = activation
    if use_dropout:
        layers_dict['dropout'] = nn.Dropout(0.3)

    return nn.Sequential(layers_dict)


class Generator(nn.Module):

    def __init__(self, z_size, conv_channels, output_size=32, kernel_size=3, stride=2,
                 normalization_block=None, use_dropout=True, init_weights=None):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()

        # Determinate optimal operations parameters
        pad = (kernel_size - 1) // stride
        current_size = output_size // (stride ** (len(conv_channels) - 1))

        # Stack all layers
        self.fc = nn.Linear(z_size, conv_channels[0] * current_size ** 2)
        self.flatten = Flatten(conv_channels[0], current_size, current_size)

        layers_dict = OrderedDict()
        for layer_idx, out_channels in enumerate(conv_channels[1:]):
            # Reverse count for ProGAN cross-stage weight loading
            naming_idx = len(conv_channels) - 1 - layer_idx
            # Add layers
            layers_dict[f"layer_{naming_idx}"] = upsampler(conv_channels[layer_idx], out_channels, kernel_size,
                                                           stride=stride, padding=pad,
                                                           normalization_block=normalization_block if layer_idx + 1 < len(conv_channels) - 1 else None,
                                                           activation=nn.ReLU() if layer_idx + 1 < len(conv_channels) - 1 else nn.Tanh(),
                                                           use_dropout=use_dropout and (layer_idx + 1 < len(conv_channels) - 1))
            # Update temp parameters
            current_size *= stride

        self.upblock = nn.Sequential(layers_dict)

        # Init weights
        if init_weights is not None:
            self.apply(init_weights)

    def freeze_layers(self, nb_layers=None):

        for name, param in self.named_parameters():
            # focus on upsampling blocks
            if name.split('.')[0].split('_')[0] == 'upblock':
                # freeze all nb_layers first layers
                if int(name.split('.')[1].split('_')[-1]) < nb_layers:
                    param.requires_grad = False

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        x = self.fc(x)
        x = self.flatten(x)
        x = self.upblock(x)

        return x
