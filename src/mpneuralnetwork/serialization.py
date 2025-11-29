import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from . import activations, layers, losses, metrics, optimizers
from .model import Model


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: int | float | NDArray) -> int | float | list:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # type: ignore[no-any-return]
        return super().default(obj)  # type: ignore[no-any-return]


def _get_class(name: str) -> Callable:
    for module in [layers, activations, losses, optimizers, metrics]:
        if hasattr(module, name):
            return cast(Callable, getattr(module, name))
    raise ValueError(f"Class {name} not found in mpneuralnetwork submodules.")


def save_model(model: Model, filepath: str) -> None:
    layers_config: list = []
    for layer in model.layers:
        layers_config.append(layer.get_config())

    optimizer_globals: dict = {}
    optimizer_params: dict = {}

    if model.optimizer.params:
        for param_name, param in model.optimizer.params.items():
            if isinstance(param, (int, float, str, bool)) or param is None:
                optimizer_globals[param_name] = param
            else:
                optimizer_params[param_name] = param

    model_config: dict = {
        "loss": model.loss.get_config(),
        "optimizer": model.optimizer.get_config(),
        "optimizer_globals": optimizer_globals,
    }

    weights_dict: dict = model.get_weights(optimizer_params)

    if filepath[-4:] != ".npz":
        filepath = f"{filepath}.npz"

    save_data: dict = weights_dict.copy()
    save_data["architecture"] = json.dumps(layers_config, cls=NumpyEncoder)
    save_data["model_config"] = json.dumps(model_config, cls=NumpyEncoder)

    np.savez_compressed(filepath, **save_data)


def load_model(path: str | Path) -> Model:
    filepath: str = str(path) if isinstance(path, Path) else path

    if filepath[-4:] != ".npz":
        filepath = f"{filepath}.npz"

    data: dict = np.load(filepath, allow_pickle=True)

    layers_config: dict = json.loads(str(data["architecture"]))
    model_config: dict = json.loads(str(data["model_config"]))

    model_layers: list = []
    for conf in layers_config:
        layer_type: str = str(conf.pop("type"))
        layer_class: Callable = _get_class(layer_type)
        layer = layer_class(**conf)
        model_layers.append(layer)

    loss_conf: dict = model_config["loss"]
    loss_type: str = str(loss_conf.pop("type"))
    loss_class: Callable = _get_class(loss_type)
    loss = loss_class(**loss_conf)

    optim_conf: dict | str = model_config["optimizer"]
    optim_type: str

    if isinstance(optim_conf, dict):
        optim_type = str(optim_conf.pop("type"))
    else:
        optim_type = str(optim_conf)
        optim_conf = {}

    optim_class: Callable = _get_class(optim_type)
    optimizer = optim_class(**optim_conf)

    if "optimizer_globals" in model_config:
        for param_name, param in model_config["optimizer_globals"].items():
            setattr(optimizer, param_name, param)

    model = Model(model_layers, loss, optimizer)
    model.restore_weights(data, optimizer)

    return model
