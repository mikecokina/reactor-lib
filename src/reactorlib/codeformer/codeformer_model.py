from __future__ import annotations

import os

import numpy as np
import torch

from PIL import Image

from ..logger import logger
from .. import settings
from ..conf.settings import EnhancementOptions
from ..logger import suppress_output
from ..modloader import load_spandrel_model
from ..restoration import CommonFaceRestoration
from ..shared import download_model


class CodeFormerCache(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CodeFormerCache, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_model"):
            self._model = None  # Initialize instance attribute

    @property
    def model(self):
        """Getter for the model attribute"""
        return self._model

    @model.setter
    def model(self, value):
        self._model = value


code_former_cache = CodeFormerCache()


class FaceRestorerCodeFormer(CommonFaceRestoration):
    def name(self):
        return "CodeFormer"

    def load_net(self) -> torch.Module:
        model_path_ = os.path.join(self.model_path, settings.FACE_RESTORATION_MODEL_DOWNLOAD_NAME)
        if not os.path.isfile(model_path_):
            download_model(model_path_, settings.FACE_RESTORATION_MODEL_URL)

        return load_spandrel_model(
            os.path.join(self.model_path, settings.FACE_RESTORATION_MODEL_DOWNLOAD_NAME),
            device=settings.device,
            expected_architecture='CodeFormer',
        ).model

    def get_device(self):
        return settings.device

    def restore(
            self,
            np_image,
            w: float | None = None,
            enhancement_options=None
    ):
        if w is None:
            w = EnhancementOptions.face_enhancement_options.codeformer_weight

        def restore_face(cropped_face_t):
            assert self.net is not None
            # noinspection PyArgumentList
            return self.net(cropped_face_t, weight=w, adain=True)[0]

        return self.restore_with_helper(
            np_image,
            restore_face,
            enhancement_options,
        )

    @staticmethod
    def setup_model(dirname: str) -> FaceRestorerCodeFormer:
        return FaceRestorerCodeFormer(dirname)


def _restore_face(
        image: Image.Image,
        instance: FaceRestorerCodeFormer,
        enhancement_options: EnhancementOptions,
) -> Image.Image:
    result_image = image

    original_image = result_image.copy()
    # noinspection PyTypeChecker
    numpy_image = np.array(result_image)
    numpy_image = instance.restore(
        numpy_image,
        w=enhancement_options.face_enhancement_options.codeformer_weight,
        enhancement_options=enhancement_options
    )
    restored_image = Image.fromarray(numpy_image)
    result_image = Image.blend(
        original_image, restored_image, enhancement_options.face_enhancement_options.codeformer_visibility
    )

    return result_image


def enhance_image(
        image: Image.Image,
        enhancement_options: EnhancementOptions,
) -> Image.Image:
    if not code_former_cache.model:
        codeformer = FaceRestorerCodeFormer.setup_model(settings.FACE_RESTORATION_MODEL_DIR)
        code_former_cache.model = codeformer

    logger.info(f"Restoring the face with CodeFormer " +
                f"(weight: {enhancement_options.face_enhancement_options.codeformer_weight})")
    with suppress_output():
        result_image = _restore_face(
            image=image,
            instance=code_former_cache.model,
            enhancement_options=enhancement_options,
        )
    return result_image
