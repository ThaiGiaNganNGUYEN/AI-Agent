"""Load packaged YAML configuration."""

from importlib.resources import files

import yaml


def load_yaml(name: str) -> dict:
    path = files("agentflow.resources").joinpath(name)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{name} must contain a mapping")
    return loaded
