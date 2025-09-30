from __future__ import annotations

import os

import numpy as np
import torch

from PIL import Image

from ..logger import logger
from .. import settings
from ..conf.settings import EnhancementOptions, FaceRestorationModels, FaceRestorer
from ..logger import suppress_output
from ..modloader import load_spandrel_model
from ..restoration import CommonFaceRestoration, restore_face, face_restoration_cache
from ..shared import download_model


class FaceRestorerCodeFormer(CommonFaceRestoration):
    @classmethod
    def name(cls):
        return "CodeFormer"

    def load_net(self) -> torch.Module:
        model_config = FaceRestorationModels.get_config(FaceRestorer.codeformer)

        model_path_ = os.path.join(self.model_path, model_config.filename)
        if not os.path.isfile(model_path_):
            download_model(model_path_, model_config.url)

        return load_spandrel_model(
            os.path.join(model_path_),
            device=settings.device,
            expected_architecture="CodeFormer",
        ).model

    def get_device(self):
        return settings.device

    def restore(
            self,
            np_image,
            enhancement_options=None,
            np_mask: np.ndarray = None
    ):
        def restore_face_executor(cropped_face_t):
            if self.net is None:
                msg = f"{self.name()} is not initialized as expected"
                raise RuntimeError(msg)

            w = enhancement_options.face_enhancement_options.restorer_weight
            # noinspection PyArgumentList
            return self.net(cropped_face_t, weight=w, adain=True)[0]

        return self.restore_with_helper(
            np_image,
            restore_face_executor,
            enhancement_options,
            np_mask
        )

    @classmethod
    def setup_model(cls, dirname: str, face_size: int = 512) -> FaceRestorerCodeFormer:
        return cls(model_path=dirname, face_size=face_size)

    @staticmethod
    def enhance_image(
            image: Image.Image,
            enhancement_options: EnhancementOptions,
            np_mask: np.ndarray = None
    ) -> Image.Image:
        return enhance_image(image, enhancement_options, np_mask)


def enhance_image(
        image: Image.Image,
        enhancement_options: EnhancementOptions,
        np_mask: np.ndarray = None
) -> Image.Image:
    is_not_current_model = face_restoration_cache.key != FaceRestorer.codeformer.value
    is_not_cached = not face_restoration_cache.model
    if is_not_cached or is_not_current_model:
        codeformer = FaceRestorerCodeFormer.setup_model(
            dirname=settings.CODEFORMER_MODEL_DIR,
            face_size=enhancement_options.face_enhancement_options.face_size
        )
        face_restoration_cache.model = codeformer
        face_restoration_cache.key = FaceRestorer.codeformer.value

    logger.info(
        "Restoring the fac with %s (weight: %s)",
        FaceRestorerCodeFormer.name(),
        enhancement_options.face_enhancement_options.restorer_weight
    )
    with suppress_output(warnings_=False, logs_=False):
        result_image = restore_face(
            image=image,
            instance=face_restoration_cache.model,
            enhancement_options=enhancement_options,
            np_mask=np_mask
        )
    return result_image
