# Adapted from https://github.com/JohnBoxAnn/TSGL-EEGNet/blob/master/core/regularizers.py#L7
# Using Claude Opus 4.5

import torch
from torch import nn


class TSG(nn.Module):
    """
    Regularizer for TSG regularization.

    Parameters
    ----------
    l1   : float, Positive L1 regularization factor.
    l21  : float, Positive L21 regularization factor.
    tl1  : float, Positive TL1 regularization factor.

    Return
    ------
    regularization : float, Regularization fine.
    """

    def __init__(self, l1: float = 0., l21: float = 0., tl1: float = 0.):
        super(TSG, self).__init__()
        self.l1 = float(l1)
        self.l21 = float(l21)
        self.tl1 = float(tl1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.l1 and not self.l21 and not self.tl1:
            return torch.tensor(0., device=x.device, dtype=x.dtype)

        regularization = torch.tensor(0., device=x.device, dtype=x.dtype)

        # Handle different input shapes to normalize to rank 3
        if x.dim() == 4 and x.shape[0] == 1:  # shape (1, len, Inputs, Outputs)
            ntf = x.squeeze(0)  # shape (len, Inputs, Outputs)
        elif x.dim() == 4 and x.shape[1] == 1:  # shape (?, 1, Timesteps, Features)
            ntf = x.squeeze(1)  # shape (?, Timesteps, Features)
        elif x.dim() == 2:  # shape (Inputs, Outputs)
            ntf = x.unsqueeze(0)  # shape (1, Inputs, Outputs)
        else:  # shape (?, Inputs, Outputs)
            ntf = x  # shape (?, Inputs, Outputs)
        # now Tensor `ntf` ranks 3

        if self.l1:
            regularization = regularization + self.l1 * torch.sum(torch.abs(ntf))

        if self.l21:
            # L21 norm: sum of L2 norms across groups
            regularization = regularization + self.l21 * torch.sum(
                torch.sqrt(
                    ntf.shape[2] * torch.sum(ntf ** 2, dim=[0, 1])
                )
            )

        if self.tl1:
            # Temporal L1: penalize differences between consecutive time steps
            regularization = regularization + self.tl1 * torch.sum(
                torch.abs(ntf[:, :-1, :] - ntf[:, 1:, :])
            )

        return regularization

    def extra_repr(self) -> str:
        return f'l1={self.l1}, l21={self.l21}, tl1={self.tl1}'
    

def tsc(tl1: float = 0.01) -> TSG:
    """
    Temporal constrained to preserve the temporal smoothness,
    for activity regularization.
    """
    return TSG(tl1=tl1)


# l1 + l2_1 = sparse group lasso
def sgl(l1: float = 0.01, l21: float = 0.01) -> TSG:
    """Sparse group lasso, for kernel regularization."""
    return TSG(l1=l1, l21=l21)


# Standard L1 regularization
class L1(nn.Module):
    """L1 regularization."""

    def __init__(self, l1: float = 0.01):
        super(L1, self).__init__()
        self.l1 = float(l1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1 * torch.sum(torch.abs(x))


# Convenience functions matching Keras API
def l1(l1_val: float = 0.01) -> L1:
    """L1 regularizer."""
    return L1(l1=l1_val)