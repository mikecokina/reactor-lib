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
from ..restoration import CommonFaceRestoration, FaceRestoration
from ..shared import download_model


codeformer: FaceRestoration | None = None


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

    def restore(self, np_image, w: float | None = None):
        if w is None:
            w = EnhancementOptions.codeformer_weight

        def restore_face(cropped_face_t):
            assert self.net is not None
            # noinspection PyArgumentList
            return self.net(cropped_face_t, weight=w, adain=True)[0]

        return self.restore_with_helper(np_image, restore_face)


def setup_model(dirname: str) -> None:
    global codeformer
    codeformer = FaceRestorerCodeFormer(dirname)


def _restore_face(image: Image, enhancement_options: EnhancementOptions):
    result_image = image
    if enhancement_options.face_restorer is not None:
        original_image = result_image.copy()
        numpy_image = np.array(result_image)
        numpy_image = codeformer.restore(
            numpy_image, w=enhancement_options.codeformer_weight
        )
        restored_image = Image.fromarray(numpy_image)
        result_image = Image.blend(
            original_image, restored_image, enhancement_options.restorer_visibility
        )

    return result_image


def enhance_image(image: Image, enhancement_options: EnhancementOptions):
    logger.info(f"Restoring the face with {codeformer.name()} (weight: {enhancement_options.codeformer_weight})")
    with suppress_output():
        result_image = image
        return _restore_face(result_image, enhancement_options)


setup_model(settings.FACE_RESTORATION_MODEL_DIR)
