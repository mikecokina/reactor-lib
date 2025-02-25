import os

import spandrel
import torch
import insightface

from .conf.settings import settings
from .inswapper.swapper import INSwapper
from .logger import suppress_output
from .reswapper.swapper import RESwapper
from .shared import download_model
from .logger import logger


def load_spandrel_model(
    path: str | os.PathLike,
    *,
    device: str | torch.device | None,
    prefer_half: bool = False,
    dtype: str | torch.dtype | None = None,
    expected_architecture: str | None = None,
) -> spandrel.ModelDescriptor:
    model_descriptor = spandrel.ModelLoader(device=device).load_from_file(str(path))
    if expected_architecture and model_descriptor.architecture != expected_architecture:
        logger.warning(
            f"Model {path!r} is not a {expected_architecture!r} model (got {model_descriptor.architecture!r})"
        )
    half = False
    if prefer_half:
        if model_descriptor.supports_half:
            model_descriptor.model.half()
            half = True
        else:
            logger.info(f"Model {path} does not support half precision, ignoring --half")
    if dtype:
        model_descriptor.model.to(dtype=dtype)
    model_descriptor.model.eval()
    logger.debug(f"Loaded {model_descriptor} from {path} (device={device}, half={half}, dtype={dtype})")
    return model_descriptor


def get_inswapper_model() -> INSwapper:
    model_path = os.path.join(settings.MODELS_PATH, settings.FACE_SWAPPER_MODEL_DOWNLOAD_NAME)
    if not os.path.isfile(model_path):
        download_model(model_path=model_path, model_url=settings.FACE_SWAPPER_MODEL_URL)
    with suppress_output(prints_=True, logs_=False, warnings_=True):
        return INSwapper(model_path=str(model_path))


def get_reswapper_model() -> RESwapper:
    model_path = str(os.path.join(settings.MODELS_PATH, settings.FACE_SWAPPER_MODEL_DOWNLOAD_NAME))
    if not os.path.isfile(model_path):
        download_model(model_path=model_path, model_url=settings.FACE_SWAPPER_MODEL_URL)
    return RESwapper(model_path=model_path, device=settings.device)


def get_analysis_model(models_path: str):
    return insightface.app.FaceAnalysis(
        name="buffalo_l",
        providers=settings.PROVIDERS,
        root=os.path.join(models_path, "insightface")
    )
