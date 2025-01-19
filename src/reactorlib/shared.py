import glob
import os
import re
from typing import Iterable, List, Union, Tuple
from pathlib import Path

import numpy as np
import torch

from . import settings
from .logger import logger


class SharedModelKeyMixin:
    """Mixin that provides _model and _key attributes and their properties."""

    def __init__(self):
        # Only initialize once per instance
        if not hasattr(self, "_model"):
            self._model = None
        if not hasattr(self, "_key"):
            self._key = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, value):
        self._key = value


class SingletonBase:
    _instances = {}

    def __new__(cls, *args, **kwargs):
        # Make sure each subclass has exactly one _instance
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonBase, cls).__new__(cls)
        return cls._instances[cls]


def get_cuda_device_string():
    if settings.DEVICE_ID is not None:
        return f"cuda:{settings.DEVICE_ID}"
    return "cuda"


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def download_model(model_path, model_url):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        logger.info(f'Downloading model to: {model_path}')
        torch.hub.download_url_to_file(model_url, model_path)


def color_generator(colors: Iterable):
    while True:
        for color in colors:
            yield color


def get_insightface_models(models_path: str):
    models_path = os.path.join(models_path, "insightface/*")
    models_ = glob.glob(models_path)
    models_ = [x for x in models_ if x.endswith(".onnx") or x.endswith(".pth")]
    return models_


def listdir(directory: str, filter_ext: List = None):
    if filter_ext is not None:
        filter_ext = set(filter_ext)

    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            to_append = None
            if filter_ext is None:
                to_append = os.path.join(root, file)
            else:
                if Path(file).suffix in filter_ext:
                    to_append = os.path.join(root, file)
            if to_append is not None:
                file_list.append(to_append)

    try:
        file_list = sorted(file_list, key=lambda x: int(re.search(r'(\d+)', Path(x).stem).group()))
    except AttributeError:
        # Skip number like ordering if not possible
        pass

    return file_list


def get_warp_affine_border_value(img: np.ndarray) -> Union[int, Tuple]:
    if img.ndim == 3:  # Color image (e.g., RGB/BGR)
        border_value = (0.0, 0.0, 0.0)
    elif img.ndim == 4:  # RGBA image
        border_value = (0.0, 0.0, 0.0, 0.0)
    else:  # Grayscale or others
        border_value = 0.0

    return border_value
