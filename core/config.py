"""Configuration-driven object construction."""

import importlib


def get_obj_from_str(string, reload=False):
    module, name = string.rsplit(".", 1)
    if reload:
        imported = importlib.import_module(module)
        importlib.reload(imported)
    return getattr(importlib.import_module(module), name)


def instantiate_from_config(config):
    if "target" not in config:
        if config in ("__is_first_stage__", "__is_unconditional__"):
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", {}))

