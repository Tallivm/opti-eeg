print('Loading imports...')

import os, argparse, logging
from datetime import datetime
import numpy as np
import torch

from models.eegnet_modular import EEGNet_Modular
from scripts import utils, train_utils, data_utils, opt_utils
from scripts.optimizers import configure_optimizer
from scripts.schedulers import configure_scheduler
from scripts.losses import get_loss_func
from scripts.eval_metrics import get_evaluation_metric

import warnings
warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths")


# TODO: try "optimal retention" strategy from https://ieeexplore.ieee.org/abstract/document/9343873


logger = logging.getLogger()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Device is: {DEVICE}')


def load_checkpoint(checkpoint_path: str, device) -> torch.nn.Module:
    logger.info(f'Loading model from checkpoint: {checkpoint_path}')    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['hyper_parameters']
    model = EEGNet_Modular(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model


def run_fold(fold_indices: tuple[list[int], list[int]], fold_i: int, config: dict,
             train_data: np.ndarray, train_labels: np.ndarray,
             test_data: np.ndarray, test_labels: np.ndarray,
             timestamp: int, disable_wandb: bool) -> dict[str, float | int]:
    
    train_dataloader, val_dataloader, test_dataloader = train_utils.create_cv_dataloaders(
        train_data=train_data, train_labels=train_labels, 
        train_val_indices=fold_indices,
        test_data=test_data, test_labels=test_labels,
        batch_size=config["params.batch_size"],
        num_workers=0, standardize=True
    )
    config["params.n_batches"] = len(train_dataloader)  # for OneCycleLR schedule

    run_name = f"fold_{fold_i}_{timestamp}"
    if not disable_wandb:
        wandb_run = wandb.init(project=config["project_name"], group=str(timestamp), config=config)        
        wandb_run.name = run_name
        wandb_run.log(config)
    else:
        wandb_run = None

    try:
        utils.set_all_seeds(config["train.model_seed"])
        model = EEGNet_Modular(**config).to(DEVICE)

        optimizer = configure_optimizer(config=config, model=model)
        scheduler = configure_scheduler(config=config, optimizer=optimizer)
        loss_fn_obj, to_argmax = get_loss_func(name=config["train.loss_func_name"])
        loss_fn = loss_fn_obj(weight=torch.tensor(config['data.class_weights']).to(DEVICE))
        eval_fn = get_evaluation_metric(name=config["train.eval_metric_name"])

        checkpoint_path = train_utils.get_run_checkpoint_name(
            run_name=run_name,
            model_name=config['params.model_name'],
            model_savedir=config['path.checkpoint_path'],
            project_name=config['project_name']
        )

        best_metrics = train_utils.train_and_val(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            eval_fn=eval_fn,
            checkpoint_path=checkpoint_path,
            wandb_run=wandb_run,
            device=DEVICE,
            to_argmax=to_argmax
        )

        best_model = load_checkpoint(checkpoint_path=checkpoint_path, device=DEVICE)
        test_loss, test_eval = train_utils.validate_epoch(
            model=best_model,
            val_dataloader=test_dataloader,
            scheduler=None,
            loss_fn=loss_fn, eval_fn=eval_fn,
            device=DEVICE, to_argmax=to_argmax,
            final_test=True
        )
        best_metrics['test_loss'] = test_loss
        best_metrics['test_eval'] = test_eval

        if wandb_run is not None:
            wandb_run.log({f"best_{k}": v for k, v in best_metrics.items()})
            wandb.finish()

        return best_metrics
    
    except Exception as e:
        if wandb_run is not None:
            wandb_run.log({'error': str(e)})
        utils.save_crash_params_dict(param_config=config, run_name=run_name)
        raise


def run_sweep_trajectory(config: dict,
                         train_data: np.ndarray, test_data: np.ndarray,
                         train_labels: np.ndarray, test_labels: np.ndarray,
                         fold_indices: list[tuple[list, list]], disable_wandb: bool) -> float:

    timestamp = int(datetime.now().timestamp())

    # run folds
    cv_metrics = {}
    for fold_i, fold in enumerate(fold_indices):
        logger.info('-'*15)
        logger.info(f'Starting fold {fold_i} for run {timestamp}...')
        logger.info('-'*15)

        fold_best_metrics = run_fold(
            fold_indices=fold, fold_i=fold_i, config=config,
            train_data=train_data, train_labels=train_labels,
            test_data=test_data, test_labels=test_labels,
            timestamp=timestamp, disable_wandb=disable_wandb
        )
        cv_metrics[fold_i] = fold_best_metrics
        logger.info(utils.str_dict_items(title=f"Best results for fold {fold_i}:", d=fold_best_metrics))

    final_val_eval = train_utils.log_trajectory_metrics(cv_metrics=cv_metrics)
    return final_val_eval


def main(trial, config: dict, disable_wandb: bool,
         train_data, train_labels, test_data, test_labels, fold_indices
         ) -> float:
    if train_data is None:
        logger.info('Loading data from scratch...')
        train_data, train_labels, test_data, test_labels = data_utils.load_and_validate_data(config=config)
        fold_indices = train_utils.get_train_val_indices_with_CV(data=train_data, labels=train_labels,
                                                                 seed=config["train"]["data_seed"], n_splits=config["train"]["n_cv"])

    if trial is None:
        opt_config = utils.read_config(config_name=config["path"]["fixed_config_path"])
    else:
        opt_config = opt_utils.create_optuna_config(trial=trial, config=config)

    static_config = opt_utils.create_static_config(config=config, data_shape=train_data.shape)
    static_config['data.class_weights'] = data_utils.get_class_weights(train_labels=train_labels)
    baseline_eval_metric = train_utils.get_baseline_eval_metric(labels=train_labels, eval_matric_name=config["train"]["eval_metric_name"])
    logger.info(f"Baseline (random) {config["train"]["eval_metric_name"]} value: {baseline_eval_metric:.2f}")
    static_config['train.eval_baseline'] = baseline_eval_metric

    static_config = opt_utils.merge_opt_with_static_config(opt_config=opt_config, static_config=static_config)

    # run trajectory
    eval_metric = run_sweep_trajectory(
        config=static_config,
        train_data=train_data,
        test_data=test_data,
        train_labels=train_labels,
        test_labels=test_labels,
        fold_indices=fold_indices,
        disable_wandb=disable_wandb
    )

    return eval_metric


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog='opti-EEG',
        description='A script to optimize hyperparameters of a chosen model, using Optuna for optimization and WanDB for logging',
    )
    parser.add_argument('configname')
    parser.add_argument('--nowandb', action='store_true', help='Flag to disable WanDB logging. Note that offline logging is not yet fully supported.')
    parser.add_argument('--noopt', action='store_true', help='Disable hyperparameter optimization entirely and instead use parameters from "fixed_params.json".')
    parser.add_argument('--loglevel', choices=['debug', 'info', 'warn'], default='info', help='Logging level: "debug", "info" (default), or "warn".')  
    args = parser.parse_args()

    utils.setup_logger(logger=logger, loglevel=args.loglevel)

    config = utils.read_config(args.configname)
    opt_utils.assert_correct_config(config, args=args)

    if '{' in config['path']['data_path']:
        raise NotImplementedError('Data file templates not yet implemented, sorry')
    else:
        dynamic_load = False
        train_data, train_labels, test_data, test_labels = data_utils.load_and_validate_data(config=config)
        fold_indices = train_utils.get_train_val_indices_with_CV(data=train_data, labels=train_labels,
                                                                 seed=config["train"]["data_seed"], n_splits=config["train"]["n_cv"])

    if args.noopt:
        main(trial=None, config=config, disable_wandb=True,
             train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels, fold_indices=fold_indices)
        logger.info('Fixed parameter run completed.')
    else:
        import optuna
        project_name = config["project_name"]

        if not args.nowandb:
            import wandb
            os.environ['WANDB_API_KEY'] = config["opt"]["wandb_user_key"]
            os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = str(config["opt"]["wandb_number_of_failures"])
            os.environ['WANDB_SILENT'] = "true"            
            wandb.login()
            
        study = optuna.create_study(
            study_name=project_name,
            storage=f"sqlite:///optuna_{project_name}.db",
            load_if_exists=True if not config["opt"]["create_new_db"] else False,
            direction="maximize" if config["train"]["eval_increases"] else "minimize",
            sampler=optuna.samplers.TPESampler()  # TODO: check other samplers
            )
        logger.info(f"Loaded Optuna study with {len(study.trials)} trials done so far...")
        study.optimize(lambda trial: main(trial=trial, config=config, disable_wandb=args.nowandb,
                                          train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels, fold_indices=fold_indices),
                       n_trials=config["opt"]["optuna_n_trials"])
        logger.info(f'Completed {config["opt"]["optuna_n_trials"]} Optuna runs.')
