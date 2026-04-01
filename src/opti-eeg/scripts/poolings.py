from typing import Callable
from torch import nn


def get_2d_pooling(name: str) -> Callable:
    if name in ('average', 'avg', 'mean'):
        return nn.AvgPool2d
    if name == 'max':
        return nn.MaxPool2d
    else:
        raise NotImplementedError(f'{name} pooling not implemented.')
    
