from typing import Callable
from torch import nn


def get_loss_func(name: str) -> tuple[Callable, bool]:
    """Returns loss function non-initialized object as well as whether it requires argmax of model output"""
    if name.lower() in ("nllloss", "nll"):
        return nn.NLLLoss, True
    if name.lower() in ("crossentropy", "crossentropyloss", "cross-entropy"):
        return nn.CrossEntropyLoss, True
    else:
        raise NotImplementedError(f"Loss '{name}' is not implemented.")