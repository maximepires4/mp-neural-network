import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from . import DTYPE, activations, layers, losses, metrics, optimizers, to_device, to_host, xp
from .layers.layer import Layer
from .optimizers import Optimizer

if TYPE_CHECKING:
    from .model import Model


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy types.

    Converts NumPy integers, floats, and arrays into native Python types
    compatible with JSON serialization.
    """

    def default(self, obj: int | float | NDArray) -> int | float | list:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # type: ignore[no-any-return]
        return super().default(obj)  # type: ignore[no-any-return]


def _get_class(name: str) -> Callable:
    """Helper to dynamically retrieve a class from a string name."""
    for module in [layers, activations, losses, optimizers, metrics]:
        if hasattr(module, name):
            return cast(Callable, getattr(module, name))
    raise ValueError(f"Class {name} not found in mpneuralnetwork submodules.")


def save_model(model: "Model", filepath: str) -> None:
    """Saves the full model state to a `.npz` archive.

    The archive contains:
    1.  `architecture` (JSON): Configuration of layers (type, size, etc.).
    2.  `model_config` (JSON): Loss, Optimizer config, and Optimizer globals (learning rate, etc.).
    3.  `layer_{i}_{param}`: Raw numpy arrays for weights and biases.
    4.  `layer_{i}_state_{name}`: Internal state (e.g., BatchNorm moving averages).
    5.  `optimizer_{param}_{layer_param}`: Optimizer state (momentum, velocity) for each parameter.

    Args:
        model (Model): The model instance to save.
        filepath (str): Destination path. If extension is missing, `.npz` is appended.
    """
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

    weights_dict: dict = get_model_weights(model.layers, optimizer_params)

    if filepath[-4:] != ".npz":
        filepath = f"{filepath}.npz"

    save_data: dict = weights_dict.copy()
    save_data["architecture"] = json.dumps(layers_config, cls=NumpyEncoder)
    save_data["model_config"] = json.dumps(model_config, cls=NumpyEncoder)

    np.savez_compressed(filepath, **save_data)


def load_model(path: str | Path) -> "Model":
    """Loads a full model from a `.npz` archive.

    This function instantiates a new `Model` object, rebuilds the layer graph,
    initializes the optimizer and loss function, and then loads all weights
    and states (including optimizer momentum) into memory.

    Args:
        path (str | Path): Path to the `.npz` file.

    Returns:
        Model: The fully restored model, ready for training or inference.
    """
    from .model import Model

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
    restore_model_weights(model.layers, data, optimizer)

    return model


def get_model_weights(layers: list[Layer], optimizer_params: dict | None = None) -> dict:
    """Extracts all weights and states from layers and optimizer.

    Args:
        layers (list[Layer]): The layers to extract from.
        optimizer_params (dict | None, optional): The specific optimizer parameters (momentums/velocities).

    Returns:
        dict: A flat dictionary mapping logical names to numpy arrays.
    """
    weights_dict: dict = {}
    for i, layer in enumerate(layers):
        if hasattr(layer, "params"):
            for l_param_name, (l_param, _) in layer.params.items():
                p_id: int = id(l_param)
                logical_name: str = f"layer_{i}_{l_param_name}"

                weights_dict[logical_name] = np.array(to_host(l_param), dtype=DTYPE, copy=True)

                if not optimizer_params:
                    continue

                for o_param_name, o_param in optimizer_params.items():
                    if not isinstance(o_param, dict):
                        continue
                    if p_id in o_param:
                        weights_dict[f"optimizer_{o_param_name}_{logical_name}"] = np.array(to_host(o_param[p_id]), dtype=DTYPE, copy=True)

        if hasattr(layer, "state"):
            for l_state_name, l_state_val in layer.state.items():
                logical_name = f"layer_{i}_state_{l_state_name}"
                weights_dict[logical_name] = np.array(to_host(l_state_val), dtype=DTYPE, copy=True)

    return weights_dict


def restore_model_weights(layers: list[Layer], weights_dict: dict, optimizer: Optimizer | None = None) -> None:
    """Restores weights and states to layers and optimizer.

    Args:
        layers (list[Layer]): The target layers to load weights into.
        weights_dict (dict): The dictionary containing loaded weights.
        optimizer (Optimizer | None, optional): The target optimizer to load state into.
    """
    for i, layer in enumerate(layers):
        if hasattr(layer, "params"):
            current_params = {}
            for l_param_name in layer.params:
                logical_name = f"layer_{i}_{l_param_name}"
                if logical_name in weights_dict:
                    current_params[l_param_name] = to_device(weights_dict[logical_name])

            if hasattr(layer, "load_params") and current_params:
                layer.load_params(current_params)

            if optimizer:
                for l_param_name, (l_param, _) in layer.params.items():
                    l_param = to_device(l_param)
                    p_id = id(l_param)
                    logical_name = f"layer_{i}_{l_param_name}"

                    for o_param_name, o_param in optimizer.params.items():
                        if not isinstance(o_param, dict):
                            continue
                        key = f"optimizer_{o_param_name}_{logical_name}"
                        if key in weights_dict:
                            if p_id not in o_param:
                                o_param[p_id] = xp.zeros_like(l_param, dtype=DTYPE)
                            xp.copyto(o_param[p_id], to_device(weights_dict[key]))  # TODO: ?

        if hasattr(layer, "state"):
            current_state = {}
            for l_state_name in layer.state:
                logical_name = f"layer_{i}_state_{l_state_name}"
                if logical_name in weights_dict:
                    current_state[l_state_name] = to_device(weights_dict[logical_name])

            if current_state:
                layer.state = current_state
