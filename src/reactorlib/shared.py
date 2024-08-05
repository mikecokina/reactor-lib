import glob
import os
from typing import Iterable, List
from pathlib import Path
import torch

from . import settings
from .logger import logger


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

    return file_list
