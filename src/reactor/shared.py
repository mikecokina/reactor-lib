import glob
import os
from typing import Iterable

import torch

from . import settings
from . logger import getLogger

logger = getLogger('reactor-lib')


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
        print(f'Downloading model to: {model_path}')
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
