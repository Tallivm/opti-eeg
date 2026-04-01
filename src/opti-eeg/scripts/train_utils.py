import operator, os, logging
from copy import deepcopy
from typing import Callable
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch import Tensor, inference_mode, tensor
from torch import from_numpy as torch_from_numpy
from torch import cat as torch_cat
from torch.nn import Module
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from scripts.utils import save_checkpoint, clean_gpu_memory, str_dict_items
from scripts.eval_metrics import get_evaluation_metric
from scripts.data_utils import standardize_by_train_3D


logger = logging.getLogger(__name__)


def get_baseline_eval_metric(labels: np.ndarray, eval_matric_name: str,
                             n_permutations: int = 100) -> float:
    eval_metric = get_evaluation_metric(name=eval_matric_name)
    permuted_labels = labels.copy()
    results = []
    for _ in range(n_permutations):
        np.random.default_rng().shuffle(permuted_labels)
        results.append(eval_metric(labels, permuted_labels))
    return float(np.mean(results))


def get_run_checkpoint_name(run_name: str, model_name: str, model_savedir: str, project_name: str, create_dirs: bool = True) -> str:
    checkpoint_dir = os.path.join(model_savedir, project_name, model_name)
    if create_dirs:
        os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{run_name}.ckpt')
    return checkpoint_path


def create_dataloader(data: np.ndarray, labels: np.ndarray, batch_size: int, num_workers: int = 0) -> DataLoader:
    return DataLoader(EEGDataset(data, labels), batch_size=batch_size, shuffle=True, num_workers=num_workers)


def create_cv_dataloaders(train_data: np.ndarray, train_labels: np.ndarray,
                          train_val_indices: tuple[list[int], list[int]],
                          test_data: np.ndarray, test_labels: np.ndarray,
                          batch_size: int, num_workers: int = 0,
                          standardize: bool = True) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_indices, val_indices = train_val_indices
    X_train, X_val = train_data[train_indices], train_data[val_indices]
    if standardize:
        X_train, X_val, X_test = standardize_by_train_3D(X_train, X_val, test_data)

    train_dataloader = create_dataloader(
        data=torch_from_numpy(X_train), labels=train_labels[train_indices],
        batch_size=batch_size, num_workers=num_workers
    )
    val_dataloader = create_dataloader(
        data=torch_from_numpy(X_val), labels=train_labels[val_indices],
        batch_size=batch_size, num_workers=num_workers
    )
    test_dataloader = create_dataloader(
        data=X_test, labels=test_labels,
        batch_size=batch_size, num_workers=num_workers)
    return train_dataloader, val_dataloader, test_dataloader


def get_train_val_indices_with_CV(data: np.ndarray, labels: np.ndarray, seed: int, n_splits: int) -> list[tuple[list, list]]:
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_indices = []
    for train_index, val_index in kfold.split(X=range(data.shape[0]), y=labels):
        fold_indices.append((train_index, val_index))
    return fold_indices


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = tensor(X).float() if not isinstance(X, Tensor) else X.float()
        self.y = tensor(y) if not isinstance(y, Tensor) else y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def initialize_xavier_uniform_weight_zero_bias(model: Module) -> None:
    """Initialize parameters of all modules with glorot uniform/xavier initialization."""

    from torch.nn.modules.batchnorm import _BatchNorm
    from torch.nn.modules.normalization import LayerNorm, GroupNorm
    from torch.nn.modules.instancenorm import _InstanceNorm
    from torch.nn.init import xavier_uniform_, constant_
    from torch.nn import Parameter

    for m in model.modules():
        if hasattr(m, "weight") and isinstance(m.weight, Parameter):
            if isinstance(m, (_BatchNorm, LayerNorm, GroupNorm, _InstanceNorm)):
                constant_(m.weight, 1)
            else:
                xavier_uniform_(m.weight, gain=1)
        if hasattr(m, "bias") and isinstance(m.bias, Parameter):
            if m.bias is not None:
                constant_(m.bias, 0)


# TRAINING

def train_batch(model: Module,
                X: Tensor, y: Tensor,
                optimizer: Optimizer, scheduler: lr_scheduler.LRScheduler,
                loss_fn: Callable, device) -> float:
    X = X.to(device)
    y = y.to(device)
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    if model.regularization_loss is not None:
        loss += model.regularization_loss(y_pred)
    train_loss = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if isinstance(scheduler, (lr_scheduler.CyclicLR, lr_scheduler.OneCycleLR)):
        scheduler.step()

    return train_loss


def train_epoch(model: Module,
                train_dataloader: DataLoader,
                optimizer: Optimizer, scheduler: lr_scheduler.LRScheduler,
                loss_fn: Callable, eval_fn: Callable,
                device, to_argmax: bool) -> tuple[float, float]:
    train_loss = 0
    model.train()

    for X, y in train_dataloader:
        batch_loss = train_batch(
            model=model, X=X, y=y, 
            optimizer=optimizer, scheduler=scheduler,
            loss_fn=loss_fn, device=device
        )
        train_loss += batch_loss

    if isinstance(scheduler, (lr_scheduler.StepLR, lr_scheduler.CosineAnnealingLR,
                              lr_scheduler.PolynomialLR, lr_scheduler.ExponentialLR,
                              lr_scheduler.LinearLR)):
        scheduler.step()

    model.eval()
    all_true, all_preds = [], []
    with inference_mode():  # Recompute train accuracy in eval mode - not needed?
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            if to_argmax:
                y_pred = y_pred.argmax(dim=1)
            all_true.append(y)
            all_preds.append(y_pred)
    model.train()

    train_loss /= len(train_dataloader)
    train_eval = eval_fn(y_true=torch_cat(all_true), y_pred=torch_cat(all_preds))
    return train_loss, train_eval


def validate_epoch(model: Module,
                   val_dataloader: DataLoader,
                   scheduler: lr_scheduler.LRScheduler | None,
                   loss_fn: Callable, eval_fn: Callable,
                   device, to_argmax: bool, final_test: bool = False) -> tuple[float, float]:
    val_loss = 0
    all_true, all_preds = [], []
    model.eval()

    with inference_mode():
        for X, y in val_dataloader:
            X, y = X.to(device), y.to(device)
            all_true.append(y)
            y_pred = model(X)
            val_loss += loss_fn(y_pred, y).item()
            if to_argmax:
                y_pred = y_pred.argmax(dim=1)
            all_preds.append(y_pred)

        if not final_test and isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

    val_loss /= len(val_dataloader)
    val_eval = eval_fn(y_true=torch_cat(all_true), y_pred=torch_cat(all_preds))
    return val_loss, val_eval


def train_and_val(model: Module, config: dict,
                  train_dataloader: DataLoader, val_dataloader: DataLoader,
                  optimizer: Optimizer, scheduler: lr_scheduler.LRScheduler,
                  loss_fn: Callable, eval_fn: Callable,
                  checkpoint_path: str, wandb_run, device, to_argmax: bool) -> dict:
   
    no_improve_count = 0
    best_log = {'val_eval': 0 if config["train.eval_increases"] else np.inf}

    for epoch in tqdm(range(config["train.max_n_epochs"])):

        if model.model_name == "EEGNetProto":
            model.compute_prototypes(train_dataloader.dataset.X.to(device),
                                     train_dataloader.dataset.y.to(device))

        train_loss, train_eval = train_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer, scheduler=scheduler,
            loss_fn=loss_fn, eval_fn=eval_fn,
            device=device, to_argmax=to_argmax,
        )

        if model.model_name == "EEGNetProto":
            model.compute_prototypes(train_dataloader.dataset.X.to(device),  # no_grad() inside so can be used during validation
                                     train_dataloader.dataset.y.to(device))

        val_loss, val_eval = validate_epoch(
            model=model,
            val_dataloader=val_dataloader,
            scheduler=scheduler,
            loss_fn=loss_fn, eval_fn=eval_fn,
            device=device, to_argmax=to_argmax,
            final_test=False
        )
        # WanDB logging
        to_log = {
            'train_loss': train_loss,
            'train_eval': train_eval,
            'val_loss': val_loss,
            'val_eval': val_eval,
            'epoch': epoch,
            'learning_rate': optimizer.param_groups[0]['lr'],
        }
        if wandb_run is not None:
            wandb_run.log({k: v for k, v in to_log.items()})

        # Early stopping
        stop, best_log, no_improve_count = check_for_early_stopping(
            train_loss=train_loss, train_eval=train_eval, val_eval=val_eval,
            best_log=best_log, no_improve_count=no_improve_count,
            model=model, optimizer=optimizer, checkpoint_path=checkpoint_path,
            current_epoch=epoch, current_log=to_log, config=config
        )
        clean_gpu_memory()
        if stop:
            break

    return best_log


def check_for_early_stopping(train_loss: float, train_eval: float, val_eval: float,
                             best_log: dict, no_improve_count: int,
                             model: Module, optimizer: Optimizer,
                             checkpoint_path: str,
                             current_epoch: int, current_log: dict, config: dict) -> tuple[str, dict, int]:
    stop_reason = None

    if np.isnan(train_loss):
        stop_reason = "loss is NaN"
        
    else:
        comparison_operator = operator.gt if config["train.eval_increases"] else operator.lt
        best_val_eval = best_log['val_eval']

        if ((current_epoch > config['train.overfit_epoch']) and comparison_operator(train_eval - config['train.overfit_thresh'], val_eval)):    
                stop_reason = "overfit"

        elif not comparison_operator(val_eval, best_val_eval + config['train.min_delta']):
            no_improve_count += 1
            if no_improve_count >= config['train.patience']:
                stop_reason = "no improvement"

        else:
            save_checkpoint(
                model=model, checkpoint_path=checkpoint_path,
                optimizer=optimizer, config=config
            )
            logger.info(f"Epoch {current_epoch}: saved checkpoint with validation metric: {val_eval:.4f}")
            no_improve_count = 0
            best_log = deepcopy(current_log)

    if stop_reason:
        logger.info(f"Early stop triggered on epoch {current_epoch}. Reason: {stop_reason}.")

    return stop_reason, best_log, no_improve_count


def get_average_metrics(metrics: dict[int, dict[str, float]], metric_names: list[str]) -> dict[str, float]:    
    average_metrics = {}
    n_folds = len(metrics)
    for k in metric_names:
        ms = [metrics[i][k] for i in range(n_folds)]
        average_metrics[f'{k}_averageof{n_folds}'] = np.mean(ms)
        average_metrics[f'{k}_sdof{n_folds}'] = np.std(ms)
    return average_metrics


def log_trajectory_metrics(cv_metrics: dict) -> float:
    logger.info('-'*20)

    average_metrics = get_average_metrics(
        cv_metrics, 
        metric_names=['train_loss', 'train_eval', 'val_loss', 'val_eval', 'test_loss', 'test_eval'])
    logger.info(str_dict_items(title=f"Average metrics for {len(cv_metrics)} folds:", d=average_metrics))
    
    avgstr = f"averageof{len(cv_metrics)}"
    average_distances = {
        f"train-val_eval_{avgstr}_distance": average_metrics[f'train_eval_{avgstr}'] - average_metrics[f'val_eval_{avgstr}'],
        f"val-test_eval_{avgstr}_distance": average_metrics[f'val_eval_{avgstr}'] - average_metrics[f'test_eval_{avgstr}']
    }
    logger.info(str_dict_items(title="Average distances between metrics:", d=average_distances))

    return average_metrics[f'val_eval_{avgstr}']
