__all__ = [
    "forgiving_true",
    "load_config",
    "log",
    "time_stamp",
]


import datetime
import pathlib
from typing import Union
import yaml


def load_config(config_path: Union[str, pathlib.Path]):
    """
    Load config and secrets
    """
    with open(config_path) as config_yaml:
        config = yaml.load(config_yaml, Loader=yaml.FullLoader)

    return config


def time_stamp():
    """
    :return: UTC time as a formatted string
    """
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H:%M:%S")


def log(message: str):
    print(f"{time_stamp()}: {message}")


def forgiving_true(expression):
    return True if expression in ("t", "True", "true", "1", 1, True) else False
