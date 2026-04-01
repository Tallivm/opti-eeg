from typing import Callable
from torch import eq as torch_eq
from sklearn.metrics import f1_score

from scripts.utils import to_numpy


def get_evaluation_metric(name: str) -> Callable:
    if name.lower() in ("acc", "accuracy"):
        return accuracy
    elif name.lower() == "macro-f1":
        return macro_f1


def accuracy(y_true, y_pred) -> float:
    correct = torch_eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100


def macro_f1(y_true, y_pred) -> float:
    return f1_score(to_numpy(y_true), to_numpy(y_pred), average='macro')
