import glob
import os
import re
from typing import Iterable, List, Union, Tuple
from pathlib import Path

import numpy as np
import torch

from .conf.settings import settings
from .logger import logger


# Define a mock tqdm class
class DummyTqdm:
    """Mock tqdm class that does nothing when progress bars are disabled."""
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self  # Allows `with` statement usage

    def __exit__(self, *args, **kwargs):
        pass  # Clean exit


class GradioTqdmWrapper:
    """Mock tqdm class that does nothing when progress bars are disabled."""
    def __init__(self, *args, **kwargs):
        self._tqdm_cls = args[0]
        self._total = None
        self._tqdm_instance = None
        self._description = None

    def __iter__(self):
        self.update()

    def update(self, *args, **kwargs):
        self._tqdm_instance.__next__()

    def __call__(self, *args, **kwargs):
        self._total = kwargs['total']
        self._description = kwargs.get("desc", "Iterate")

        self._tqdm_instance = self._tqdm_cls(range(self._total), desc=self._description)
        return self

    def __enter__(self):
        return self  # Allows `with` statement usage

    def __exit__(self, *args, **kwargs):
        pass  # Clean exit


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


def get_tqdm_cls(progressbar: bool = True):
    # Try importing tqdm only if progress bar is enabled
    try:
        if progressbar:
            from tqdm import tqdm
        else:
            raise ImportError()
    except ImportError:
        tqdm = DummyTqdm  # Use mock progress bar

    return tqdm


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


def listdir(directory: str, filter_ext: List = None) -> List[str]:
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


def get_bytes_from_url(path) -> bytes:
    import requests

    if path.startswith('/'):
        with open(path, 'rb') as f:
            image_bytes = f.read()
    else:
        resp = requests.get(path)
        assert resp.ok
        image_bytes = resp.content
    return image_bytes


def bbox_percentage(x1, y1, x2, y2, w, h):
    # Compute the area of the bounding box
    bbox_area = abs(x2 - x1) * abs(y2 - y1)

    # Compute the area of the image
    image_area = w * h

    # Calculate the percentage of the image covered by the bbox
    percentage = (bbox_area / image_area) * 100
    return percentage
