def suggest_names(trial, sample_rate: int) -> dict:
    return {
        "model_name": trial.suggest_categorical("model_name", ["EEGNet", "EEGNetConv", "EEGNetProto", "TSGLEEGNet"]),
        "optimizer_name": trial.suggest_categorical("optimizer_name", ["sgd", "adam", "adamw"]),
        "scheduler_name": trial.suggest_categorical("scheduler_name", ["step", "cosine", "cyclic", "linear", "exponential", "polynomial", "cycle", "none"]),
        "depthwise_activation_name": trial.suggest_categorical("depthwise_activation_name", ["mish", "relu", "elu", "gelu", "leaky_relu", "selu", "celu"]),
        "separable_activation_name": trial.suggest_categorical("separable_activation_name", ["mish", "relu", "elu", "gelu", "leaky_relu", "selu", "celu"]),
        "depthwise_pool_name": trial.suggest_categorical("depthwise_pool_name", ["mean", "max"]),
        "separable_pool_name": trial.suggest_categorical("separable_pool_name", ["mean", "max"]),

        "batch_size": trial.suggest_int("batch_size", 32, 512),
        "F1": trial.suggest_int("F1", 4, 32),
        "F2": trial.suggest_int("F2", 4, 32),
        "D": trial.suggest_int("D", 2, 32),
        "depthwise_pool_width": trial.suggest_int("depthwise_pool_width", 2, 6),
        "separable_pool_width": trial.suggest_int("separable_pool_width", 4, 8),
        "temporal_kernel_length": trial.suggest_int("temporal_kernel_length", sample_rate//4, sample_rate),
        "separable_kernel_length": trial.suggest_int("separable_kernel_length", 4, 32),
        "depthwise_dropout_rate": trial.suggest_float("depthwise_dropout_rate", 0.1, 0.6),
        "separable_dropout_rate": trial.suggest_float("separable_dropout_rate", 0.1, 0.6),
        "depthwise_dropout_name": trial.suggest_categorical("depthwise_dropout_name", ['simple', '2d']),
        "separable_dropout_name": trial.suggest_categorical("separable_dropout_name", ['simple', '2d']),
        "conv_depth_max_norm": trial.suggest_float("conv_depth_max_norm", 0.8, 1.0),

        "temporal_momentum": trial.suggest_float("temporal_momentum", 0.001, 0.01),
        "depthwise_momentum": trial.suggest_float("depthwise_momentum", 0.001, 0.01),
        "separable_momentum": trial.suggest_float("separable_momentum", 0.001, 0.01),
        "temporal_eps": trial.suggest_float("temporal_eps", 1e-4, 1e-2, log=True),
        "depthwise_eps": trial.suggest_float("depthwise_eps", 1e-4, 1e-2, log=True),
        "separable_eps": trial.suggest_float("separable_eps", 1e-4, 1e-2, log=True),
        "temporal_affine": trial.suggest_categorical("temporal_affine", [True, False]),
        "depthwise_affine": trial.suggest_categorical("depthwise_affine", [True, False]),
        "separable_affine": trial.suggest_categorical("separable_affine", [True, False]),
    }


def scheduler_params(name: str, trial, n_epochs: int) -> dict:
    # TODO: add "plateau"
    if name == "linear":
        return {
            "scheduler_start_factor": trial.suggest_float("scheduler_start_factor", 0.4, 1.0),
            "scheduler_end_factor": trial.suggest_float("scheduler_end_factor", 0.2, 1.0),
            "scheduler_total_iters": trial.suggest_int("scheduler_total_iters", 5, n_epochs)
        }
    if name == "exponential":
        return {
            "scheduler_gamma": trial.suggest_float("scheduler_gamma", 0.05, 1.0),
        }
    if name == "polynomial":
        return {
            "scheduler_total_iters": trial.suggest_int("scheduler_total_iters", 5, n_epochs),
            "scheduler_polynomial_power": trial.suggest_float("scheduler_polynomial_power", 0.05, 5),
        }
    if name == "step":
        return {
            "scheduler_gamma": trial.suggest_float("scheduler_gamma", 0.05, 1.0),
            "scheduler_step_size": trial.suggest_int("scheduler_step_size", 5, 30),
        }
    if name == "cosine":
        return {
            "scheduler_eta_min": trial.suggest_float("scheduler_eta_min", 1e-6, 1e-4, log=True),
            "scheduler_total_iters": trial.suggest_int("scheduler_total_iters", 5, n_epochs),
        }
    if name == "cyclic":
        return {
            "scheduler_base_lr": trial.suggest_float("scheduler_base_lr", 1e-6, 1e-2, log=True),
            "scheduler_max_lr": trial.suggest_float("scheduler_max_lr", 1e-6, 1e-2, log=True),
            "scheduler_step_size": trial.suggest_int("scheduler_step_size", 5, 30),
        }
    if name == "cycle":
        return {            
            "scheduler_base_lr": trial.suggest_float("scheduler_base_lr", 1e-6, 1e-2, log=True),
            "scheduler_max_lr": trial.suggest_float("scheduler_max_lr", 1e-6, 1e-2, log=True),
            "scheduler_cycle_final_div_factor": trial.suggest_float("scheduler_cycle_final_div_factor", 50, 10000, log=True),
            "scheduler_pct_start": trial.suggest_float("scheduler_pct_start", 0.01, 0.2),
            "scheduler_pct_training": trial.suggest_float("scheduler_pct_training", 0.8, 1.0),
            "scheduler_cycle_three_phase": trial.suggest_categorical("scheduler_cycle_three_phase", [True, False]),
        }
    if name == "none":
        return {}
    else:
        raise ValueError(f'Unknown scheduler name: {name}')


def optimizer_params(name: str, trial) -> dict:
    if name == "sgd":
        return {
            "optimizer_lr": trial.suggest_float("optimizer_lr", 1e-4, 1e-2, log=True),
            "optimizer_momentum": trial.suggest_float("optimizer_momentum", 0.1, 0.9),
            "optimizer_weight_decay": trial.suggest_float("optimizer_weight_decay", 1e-4, 1e-3, log=True),           
        }
    if name in ["adam", "adamw"]:
        return {
            "optimizer_lr": trial.suggest_float("optimizer_lr", 1e-4, 1e-2, log=True),
            "optimizer_beta1": trial.suggest_float("optimizer_beta1", 0.6, 0.9),
            "optimizer_beta2": trial.suggest_float("optimizer_beta2", 0.75, 0.999),
            "optimizer_eps": trial.suggest_float("optimizer_eps", 1e-7, 1e-6, log=True), 
            "optimizer_weight_decay": trial.suggest_float("optimizer_weight_decay", 1e-4, 1e-3, log=True), 
        }
    else:
        raise ValueError(f'Unknown optimizer name: {name}')
    

def model_params(name: str, trial) -> dict:
    if name == "EEGNet":
        return {
            "linear_max_norm": trial.suggest_float("linear_max_norm", 0.2, 0.8),
        }
    if name == "TSGLEEGNet":
        return {
            "l1": trial.suggest_float("l1", 1e-5, 1e-3, log=True),
            "l21": trial.suggest_float("l21", 1e-5, 1e-3, log=True),
            "tl1": trial.suggest_float("tl1", 1e-4, 1e-2, log=True),
            "linear_max_norm": trial.suggest_float("linear_max_norm", 0.2, 0.8),
        }
    if name == "EEGNetProto":
        return {
            "proto_temperature": trial.suggest_float("proto_temperature", 0.5, 2.0),
        }
    if name == "EEGNetConv":
        return {}
    else:
        raise ValueError(f'Unknown model name: {name}')