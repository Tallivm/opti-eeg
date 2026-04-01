from torch import optim
from torch.nn import Module

from scripts.utils import assert_required_params


def configure_optimizer(config: dict, model: Module) -> optim.Optimizer:
    optimizer_config = {k.split('params.', 1)[-1]: v for k, v in config.items() if k.startswith('params.optimizer')}
    return get_optimizer(name=optimizer_config['optimizer_name'], model=model, **optimizer_config)


def get_optimizer(name: str, model: Module, **kwargs) -> optim.Optimizer:
    if name == "sgd":
        assert_required_params(["optimizer_lr", "optimizer_momentum", "optimizer_weight_decay"], kwargs, name)
        return optim.SGD(
            model.parameters(), 
            lr=kwargs["optimizer_lr"], 
            momentum=kwargs["optimizer_momentum"],
            weight_decay=kwargs["optimizer_weight_decay"]
        )
    if name == "adamw":
        assert_required_params(["optimizer_lr", "optimizer_beta1", "optimizer_beta2", 
                                "optimizer_eps", "optimizer_weight_decay"], kwargs, name)
        return optim.AdamW(
            model.parameters(), 
            lr=kwargs["optimizer_lr"], 
            betas=(kwargs["optimizer_beta1"], kwargs["optimizer_beta2"]),
            eps=kwargs["optimizer_eps"],
            weight_decay=kwargs["optimizer_weight_decay"]
        )
    if name == "adam":
        assert_required_params(["optimizer_lr", "optimizer_beta1", "optimizer_beta2", 
                                "optimizer_eps", "optimizer_weight_decay"], kwargs, name)
        return optim.Adam(
            model.parameters(), 
            lr=kwargs["optimizer_lr"], 
            betas=(kwargs["optimizer_beta1"], kwargs["optimizer_beta2"]),
            eps=kwargs["optimizer_eps"],
            weight_decay=kwargs["optimizer_weight_decay"]
        )
    else:
        raise NotImplementedError(f'{name} optimizer not implemented.')