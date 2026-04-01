from torch.optim import lr_scheduler, Optimizer

from scripts.utils import assert_required_params


def configure_scheduler(config: dict, optimizer: Optimizer) -> lr_scheduler.LRScheduler | None:
    scheduler_config = {k.split('params.', 1)[-1]: v for k, v in config.items() if k.startswith('params.scheduler')}
    scheduler_config['max_train_epochs'] = config['train.max_n_epochs']
    scheduler_config['n_batches'] = config['params.n_batches']
    return get_scheduler(name=scheduler_config['scheduler_name'], optimizer=optimizer, **scheduler_config)


def get_scheduler(name: str, optimizer: Optimizer, **kwargs) -> lr_scheduler.LRScheduler | None:
    if name == "linear":
        assert_required_params(["scheduler_start_factor", "scheduler_end_factor", "scheduler_total_iters"], kwargs, name)
        return lr_scheduler.LinearLR(
            optimizer, 
            start_factor=kwargs["scheduler_start_factor"],
            end_factor=kwargs["scheduler_end_factor"],
            total_iters=kwargs["scheduler_total_iters"]
        )
    if name == "exponential":
        assert_required_params(["scheduler_gamma"], kwargs, name)
        return lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=kwargs["scheduler_gamma"]
        )
    if name == "polynomial":
        assert_required_params(["scheduler_total_iters", "scheduler_polynomial_power"], kwargs, name)
        return lr_scheduler.PolynomialLR(
            optimizer, 
            total_iters=kwargs["scheduler_total_iters"],
            power=kwargs["scheduler_polynomial_power"]
        )
    if name == "step":
        assert_required_params(["scheduler_step_size", "scheduler_gamma"], kwargs, name)
        return lr_scheduler.StepLR(
            optimizer, 
            step_size=kwargs["scheduler_step_size"],
            gamma=kwargs["scheduler_gamma"]
        )
    if name == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=kwargs["max_train_epochs"]
        )
    if name == "plateau":
        assert_required_params(["scheduler_gamma", "scheduler_patience", "scheduler_plateau_threshold"], kwargs, name)
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=kwargs["scheduler_gamma"], 
            patience=kwargs["scheduler_patience"],
            threshold=kwargs["scheduler_plateau_threshold"]
        )
    if name == "cyclic":
        assert_required_params(["scheduler_base_lr", "scheduler_max_lr", "scheduler_step_size", "scheduler_step_size"], kwargs, name)
        return lr_scheduler.CyclicLR(
            optimizer, 
            base_lr=kwargs['scheduler_base_lr'], 
            max_lr=kwargs['scheduler_max_lr'],
            step_size_up=kwargs['scheduler_step_size'],
            step_size_down=kwargs['scheduler_step_size'],
            cycle_momentum=False,
            mode='triangular',
            scale_fn=None,
            scale_mode='cycle'
        )
    if name == "cycle":
        assert_required_params(["scheduler_pct_training", "scheduler_pct_start", "scheduler_cycle_three_phase",
                                "scheduler_base_lr", "scheduler_max_lr", "scheduler_cycle_final_div_factor"], kwargs, name)
        div_factor = kwargs['scheduler_max_lr'] / kwargs['scheduler_base_lr']
        final_div_factor = kwargs['scheduler_cycle_final_div_factor']
        return lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=kwargs['scheduler_max_lr'], 
            total_steps=None,
            steps_per_epoch=kwargs['n_batches'],
            epochs=kwargs['max_train_epochs'],
            pct_start=kwargs['scheduler_pct_start'],
            three_phase=kwargs["scheduler_cycle_three_phase"],
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            anneal_strategy='cos'
        )
    if name == "none":
        return None
    else:
        raise NotImplementedError(f'{name} scheduler not implemented.')
    