from torch import nn
from scripts.custom_nn_modules import Conv2dWithConstraint, SeparableConv2d, LinearWithConstraint
from scripts.activations import get_activation
from scripts.poolings import get_2d_pooling


def EEGNet_temporal_conv_with_batchnorm(F1: int, kernel_length: int,
                                        momentum: float, affine: bool, eps: float,
                                        same_padding: bool = True) -> nn.ModuleDict:
    padding = 'same' if same_padding else (0, kernel_length // 2)
    modules = nn.ModuleDict({
        'temporal_conv': nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=padding,
            bias=False
        ),
        'temporal_batchnorm': nn.BatchNorm2d(F1, momentum=momentum, affine=affine, eps=eps)
    })
    return modules

def EEGNet_depthwise_conv_with_batchnorm(F1: int, D: int, n_channels: int, max_norm: float,
                                         momentum: float, affine: bool, eps: float) -> nn.ModuleDict:
    modules = nn.ModuleDict({
        'depthwise_conv': Conv2dWithConstraint(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,  # This makes it depthwise
            max_norm=max_norm,
            padding=0,
            bias=False
        ),
        'depthwise_batchnorm': nn.BatchNorm2d(F1 * D, momentum=momentum, affine=affine, eps=eps)
    })
    return modules

def EEGNet_separable_conv_with_batchnorm(F1: int, F2: int, D: int, kernel_length: int,
                                         momentum: float, affine: bool, eps: float,
                                         same_padding: bool = True) -> nn.ModuleDict:
    padding = 'same' if same_padding else (0, kernel_length // 2)
    modules = nn.ModuleDict({
        'separable_conv': SeparableConv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, kernel_length),
            padding=padding,
            bias=False
        ),
        'separable_batchnorm': nn.BatchNorm2d(F2, momentum=momentum, affine=affine, eps=eps)
    })
    return modules


def EEGNet_TSGL_simple_cov_with_batchnorm(F1: int, F2: int, D: int, kernel_length: int,
                                          momentum: float, affine: bool, eps: float,
                                          same_padding: bool = True) -> nn.ModuleDict:
    padding = 'same' if same_padding else (0, kernel_length // 2)
    modules = nn.ModuleDict({
        'tsgl_simple_conv': nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, kernel_length),
            padding=padding,
            bias=False
        ),
        'tsgl_batchnorm': nn.BatchNorm2d(F2, momentum=momentum, affine=affine, eps=eps)
    })
    return modules


def EEGNet_simple_classifier(F2: int, final_dim_size: int, n_classes: int, max_norm: float) -> nn.ModuleDict:
    modules = nn.ModuleDict({
        'linear_classifier_flatten': nn.Flatten(),
        'linear_classifier': LinearWithConstraint(F2 * final_dim_size, out_features=n_classes, max_norm=max_norm)
    })
    return modules

def EEGNet_conv_classifier(F2: int, final_dim_size: int, n_classes: int) -> nn.ModuleDict:
    modules = nn.ModuleDict({
        'conv_classifier': nn.Conv2d(
            in_channels=F2,
            out_channels=n_classes,
            kernel_size=(1, final_dim_size),
            bias=True
        ),
        'conv_classifier_flatten': nn.Flatten(),
    })
    return modules    


def activation_pool_dropout(activation_name: str, pool_name: str, pool_kernel_width: int,
                            dropout_name: str, dropout_rate: float, prefix: str) -> nn.ModuleDict:
    modules = nn.ModuleDict({
        f'{prefix}_activation': get_activation(activation_name),
        f'{prefix}_pool': get_2d_pooling(pool_name)(kernel_size=(1, pool_kernel_width), ceil_mode=False),
        f'{prefix}_dropout': _get_dropout(dropout_name)(dropout_rate)
    })
    return modules


def _get_dropout(dropout_name: str) -> nn.Module:
    if dropout_name.lower() in ['simple', 'dropout']:
        return nn.Dropout
    elif dropout_name.lower() in ['2d', 'dropout2d']:
        return nn.Dropout2d
