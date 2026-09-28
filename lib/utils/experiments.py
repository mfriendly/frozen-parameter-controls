import os
from typing import Optional, Mapping

import math
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers import NeptuneLogger, WandbLogger
from tsl.experiment import Experiment


def min_conv_layers(window: int,
                    tcn_params: dict,
                    n_outer_layers: int = 1) -> int:
    """
    Computes the minimum number of convolutional layers per outer layer
    required to cover the entire receptive field of size `window`.

    Parameters:
    - window (int): The target receptive field size to cover.
    - kernel_size (int): The kernel size of the convolution.
    - n_outer_layers (int): The number of outer layers (L). Default is 1.
    - dilation (int): The base dilation factor (D). Default is 1.
    - exponential_dilation (bool): If True, assumes dilation grows as D^i.
                                   If False, assumes constant dilation.

    Returns:
    - int: Minimum number of convolutional layers per outer layer (L_t).
    """
    tcn_params = dict(**tcn_params)
    dilation = tcn_params.get('dilation', 1)
    exponential_dilation = tcn_params.get('exponential_dilation', True)
    dilation_mod = tcn_params.get('dilation_mod')
    kernel_size = tcn_params.get(
        'kernel_size', tcn_params.get('temporal_kernel_size'))

    if kernel_size <= 1 or window <= 1 or n_outer_layers <= 0 or dilation < 1:
        raise ValueError("Invalid input values: ensure kernel_size > 1, window > 1, "
                         "n_outer_layers > 0, dilation >= 1")

    if dilation_mod is not None:
        assert exponential_dilation, \
            "dilation_mod is only valid for exponential dilation"
        tcn_layers = 0
        rf = 1
        while rf < window:
            tcn_layers += 1
            rf = tcn_rf(tcn_params, tcn_layers, n_outer_layers)
        return tcn_layers

    if exponential_dilation and dilation > 1:
        # Exponential dilation case (D^i)
        numerator = (window - 1) * (dilation - 1)
        denominator = (kernel_size - 1) * n_outer_layers
        if denominator <= 0:
            raise ValueError("Invalid parameters: (kernel_size - 1) * n_outer_layers "
                             "must be positive.")

        L_t = math.log(numerator / denominator + 1, dilation)
    else:
        # Fixed dilation case (D constant)
        denominator = (kernel_size - 1) * dilation * n_outer_layers
        if denominator <= 0:
            raise ValueError("Invalid parameters: (kernel_size - 1) * dilation * "
                             "n_outer_layers must be positive.")

        L_t = (window - 1) / denominator

    return math.ceil(L_t)  # Round up since L_t must be an integer


def tcn_rf(tcn_params: dict, tcn_layers: int, n_outer_layers: int = 1) -> int:
    dilation = d = tcn_params.get('dilation', 1)
    exponential_dilation = tcn_params.get('exponential_dilation', True)
    dilation_mod = tcn_params.get('dilation_mod')
    kernel_size = tcn_params.get('kernel_size',
                                 tcn_params.get('temporal_kernel_size'))

    rf = 1
    for i in range(n_outer_layers):
        for l_t in range(tcn_layers):
            if exponential_dilation:
                exp = l_t if dilation_mod is None else l_t % dilation_mod
                d = dilation ** exp
            rf += d * (kernel_size - 1)
    return rf


def get_spatial_order(n_nodes, graph_generator) -> int:
    if graph_generator == "path":
        return n_nodes - 1
    if graph_generator in ["ring", "crossed_ring"]:
        return n_nodes // 2
    if graph_generator == "lollipop":
        return n_nodes // 2 + 1
    if graph_generator == 'barbell':
        return 3
    if graph_generator.endswith('barbell'):
        path_len = int(graph_generator.split('_')[0])
        return path_len + 2


# Register custom OmegaConf resolvers on import
def register_custom_resolvers():
    OmegaConf.register_new_resolver(name='fdiv', resolver=lambda x, d: x // d)
    OmegaConf.register_new_resolver(name='cdiv', resolver=lambda x, d: math.ceil(x / d))
    OmegaConf.register_new_resolver(name='floor', resolver=lambda x: int(x))
    OmegaConf.register_new_resolver(name='ceil', resolver=lambda x: math.ceil(x))
    OmegaConf.register_new_resolver(name='eq', resolver=lambda x, y: x == y)
    OmegaConf.register_new_resolver(name='leq', resolver=lambda x, y: x <= y)
    OmegaConf.register_new_resolver(name='geq', resolver=lambda x, y: x >= y)
    OmegaConf.register_new_resolver(name='gt', resolver=lambda x, y: x > y)
    OmegaConf.register_new_resolver(name='lt', resolver=lambda x, y: x < y)
    OmegaConf.register_new_resolver(name='max', resolver=lambda *args: max(args))
    OmegaConf.register_new_resolver(name='log', resolver=lambda x, b: math.log(x, b))
    OmegaConf.register_new_resolver(name='min_conv_layers', resolver=min_conv_layers)
    OmegaConf.register_new_resolver(name='tcn_rf', resolver=tcn_rf)
    OmegaConf.register_new_resolver(name='spatial_order', resolver=get_spatial_order)


register_custom_resolvers()


def nest_dict(d: Mapping, sep: str = '/') -> dict:
    """
    Transforms a dictionary with :attr:`sep` separated keys into a nested dictionary.

    Args:
        d (dict): A dictionary where keys are strings containing :attr:`sep` separated
            paths.
        sep (str): The separator used to split the keys. (default: :obj:`'/'`)

    Returns:
        dict: A nested dictionary where '/' separated keys are converted into nested
            keys.

    Example:
        >>> d = {'a/b/c': 1, 'a/b/d': 2, 'x/y': 3}
        >>> nest_dict(d)
        {
            'a': {
                'b': {
                    'c': 1,
                    'd': 2
                }
            },
            'x': {
                'y': 3
            }
        }
    """

    def insert(keys, value):
        if len(keys) == 1:
            return {keys[0]: value}
        return {keys[0]: insert(keys[1:], value)}

    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        temp = insert(parts, value)
        for k, v in temp.items():
            result = {**result, k: {**result.get(k, {}), **v}}
    return result


def update_nested_dict(original, updates):
    """
    Recursively update a nested dictionary with another dictionary.

    Parameters:
    - original (dict): The original dictionary to be updated.
    - updates (dict): The dictionary containing updates.

    Returns:
    - dict: The updated dictionary.
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in original and isinstance(original[key],
                                                                      dict):
            # Recursively update nested dictionaries
            original[key] = update_nested_dict(original[key], value)
        else:
            # Update or add the value
            original[key] = value
    return original


def get_logger(exp: Experiment, cfg: DictConfig, **run_args) -> Optional[Logger]:
    if cfg.get('logger') is None:
        return None

    exp_args = exp.get_config_dict()
    update_nested_dict(exp_args, nest_dict(run_args, sep='__'))

    exp_name = cfg.run.name

    if cfg.logger.backend == 'wandb':
        entity, project = cfg.logger.project.split('/')
        import wandb
        wandb.finish()  # to be called to not log on the same run as the previous one
        exp_logger = WandbLogger(name=exp_name,
                                 save_dir=cfg.run.dir,
                                 offline=cfg.logger.offline,
                                 entity=entity,
                                 project=project,
                                 config=exp_args,
                                 tags=cfg.tags)
    elif cfg.logger.backend == 'neptune':
        from neptune.utils import stringify_unsupported
        exp_args = stringify_unsupported(exp_args)
        exp_logger = NeptuneLogger(project_name=cfg.neptune.project,
                                   experiment_name=exp_name,
                                   save_dir=cfg.run.dir,
                                   tags=cfg.tags,
                                   params=exp_args,
                                   debug=cfg.neptune.offline)
        exp_logger.log_artifact(os.path.join(exp.run_dir, "config.yaml"))
    else:
        raise ValueError(f"Logger {cfg.logger.backend} not available.")
    return exp_logger


def finalize_experiment(logger: Optional[Logger]):
    if logger is None:
        return
    logger.finalize('success')
    if isinstance(logger, WandbLogger):
        import wandb
        wandb.finish()
