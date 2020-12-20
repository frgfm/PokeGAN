from functools import partial
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from typing import List, Callable, Any, Tuple, Union, Optional


__all__ = ['Discriminator', 'Generator']


class Discriminator(nn.Sequential):
    def __init__(
        self,
        img_size: int,
        conv_chans: List[int],
        k_size: int,
        init_params: Optional[Callable[[nn.Module], Any]] = None,
        dropout: float = 0.,
        bn: bool = True
    ) -> None:

        _layers: List[nn.Module] = []
        # Conv layers
        for in_chan, out_chan in zip(conv_chans[:-1], conv_chans[1:]):
            _block: List[nn.Module] = []
            _block.append(nn.Conv2d(in_chan, out_chan, k_size, stride=2, padding=k_size // 2, bias=not bn))
            if bn:
                _block.append(nn.BatchNorm2d(out_chan))
            _block.append(nn.LeakyReLU(0.2))
            if dropout > 0:
                _block.append(nn.Dropout(dropout))
            _layers.append(nn.Sequential(*_block))

        super().__init__(
            nn.Sequential(*_layers),
            nn.Flatten(1),
            nn.Linear(int(conv_chans[-1] * (img_size / 2 ** (len(conv_chans) - 1)) ** 2), 1),
        )

        if init_params is None:
            init_params = partial(default_init_params, nonlinearity='leaky_relu')

        self.apply(init_params)


class UnFlatten(nn.Module):
    def __init__(self, shape: Union[List[int], Tuple[int, ...]]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, *self.shape)


class Generator(nn.Sequential):
    def __init__(
        self,
        z_size: int,
        img_size: int,
        conv_chans: List[int],
        k_size: int,
        init_params: Optional[Callable[[nn.Module], Any]] = None,
        dropout: float = 0.,
        bn: bool = True
    ) -> None:

        _layers: List[nn.Module] = []
        # FC layer
        _reduced_size = int(img_size / 2 ** (len(conv_chans) - 1))
        # Conv layers
        _idx = 1
        for in_chan, out_chan in zip(conv_chans[:-1], conv_chans[1:]):
            _block: List[nn.Module] = []
            _block.append(nn.ConvTranspose2d(in_chan, out_chan, k_size, stride=2, padding=k_size // 2,
                                             output_padding=1, bias=(not bn) and (_idx < len(conv_chans) - 1)))
            if bn and _idx < len(conv_chans) - 1:
                _block.append(nn.BatchNorm2d(out_chan))
            if _idx == len(conv_chans) - 1:
                _block.append(nn.Tanh())
            else:
                _block.append(nn.LeakyReLU(0.2))
            if dropout > 0 and _idx < len(conv_chans) - 1:
                _block.append(nn.Dropout(dropout))
            _layers.append(nn.Sequential(*_block))
            _idx += 1

        super().__init__(
            nn.Linear(z_size, conv_chans[0] * _reduced_size ** 2),
            UnFlatten((conv_chans[0], _reduced_size, _reduced_size)),
            nn.Sequential(*_layers),
        )

        if init_params is None:
            init_params = partial(default_init_params, nonlinearity='leaky_relu')

        self.apply(init_params)
        last_conv_idx = len(self[-1]) - 2
        getattr(self[-1], str(last_conv_idx)).apply(partial(init_params, nonlinearity='tanh'))


def default_init_params(m: nn.Module, nonlinearity: str = 'leaky_relu') -> None:
    """Initializes pytorch modules

    Args:
        module: module to initialize
        nonlinearity: linearity to initialize convolutions for
    """

    if isinstance(m, _ConvNd):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity=nonlinearity)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
