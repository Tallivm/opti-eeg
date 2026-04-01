from typing import Callable
from torch import nn


def get_activation(name: str) -> Callable:
    nn_activations = {
        'elu': nn.ELU(),
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'selu': nn.SELU(),
        'celu': nn.CELU(),
        'mish': nn.Mish(),
        'leaky_relu': nn.LeakyReLU()
    }
    if name in nn_activations.keys():
        return nn_activations[name]
    else:
        raise NotImplementedError(f'{name} activation not implemented.')
    