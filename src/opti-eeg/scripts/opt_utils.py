import os
from flatten_dict import flatten as flatten_dict

import optuna_params


def assert_correct_config(config: dict, args) -> None:
    # assert that paths exist
    assert os.path.exists(config["path"]["data_path"]), f'Data path does not exist: {config["path"]["data_path"]}'
    assert os.path.exists(config["path"]["labels_path"]), f'Labels path does not exist: {config["path"]["labels_path"]}'
    if config["path"]["test_data_path"]:
        assert os.path.exists(config["path"]["test_data_path"]), "Test data path does not exist!"
        assert config["path"]["test_labels_path"], "If test data path provided, test labels path should be provided as well!"
        assert os.path.exists(config["path"]["test_labels_path"]), "Test labels path does not exist!"
    if args.noopt:
        assert os.path.exists(config['path']["fixed_config_path"]), "Optimization disabled, but parameter config not found."
    # assert correct params
    assert config["data"]["n_classes"] == len(config["data"]["class_names"]), "n_classes and length of class names do not match!"
    assert config["data"]["n_classes"] > 1, "Number of classes should be at least 2!"
    assert config["data"]["n_channels"] == len(config["data"]["channel_names"]), "n_channels and length of channel names do not match!"
    assert config["data"]["n_channels"] > 0, "Number of channels should be at least 1!"
    assert all(x in config["data"]["channel_names"] for x in config["data"]["omit_channels"]), "One or more channels to omit are not in the list of channels!"
    assert len(config["data"]["omit_channels"]) < len(config["data"]["channel_names"]), "Too many channels are omitted, no data left!"
    if config["data"]["lowpass"] and config["data"]["highpass"]:
        assert config["data"]["lowpass"] > config["data"]["highpass"], "Highpass Hz should be lower than lowpass Hz!"
    # other asserts
    if (not args.noopt) and (not args.nowandb) and not (args.nowandb):
        assert config["opt"]["wandb_user_key"], "To connect to WanDB, provide user key."


def create_static_config(config: dict, data_shape: tuple[int, int, int]) -> dict:
    static_config = flatten_dict(config, reducer='dot')
    static_config['data.n_channels'] = data_shape[1]
    static_config['data.n_timestamps'] = data_shape[2]
    return static_config


def create_optuna_config(trial, config: dict) -> dict:
    optuna_config = optuna_params.suggest_names(trial=trial, sample_rate=config["data"]['sample_rate'])
    model_config = optuna_params.model_params(name=optuna_config['model_name'], trial=trial)
    optuna_config.update(model_config)
    optimizer_config = optuna_params.optimizer_params(name=optuna_config['optimizer_name'], trial=trial)
    optuna_config.update(optimizer_config)
    scheduler_config = optuna_params.scheduler_params(name=optuna_config['scheduler_name'], trial=trial,
                                                      n_epochs=config["train"]["max_n_epochs"])
    optuna_config.update(scheduler_config)
    return optuna_config


def merge_opt_with_static_config(opt_config: dict, static_config: dict) -> dict:
    for k, v in opt_config.items():
        static_config[f"params.{k}"] = v
    return static_config
