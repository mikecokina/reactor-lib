from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from torch import nn

from . import GFPGANv1Clean
from ..conf.settings import FaceRestorationModels, FaceRestorer, EnhancementOptions
from ..restoration import CommonFaceRestoration, face_restoration_cache, restore_face
from .. import settings
from ..shared import download_model
from ..logger import logger, suppress_output


class FaceRestorerGFPGAN(CommonFaceRestoration):
    @classmethod
    def name(cls):
        return "GFPGAN"

    def load_net(self) -> nn.Module:
        model_config = FaceRestorationModels.get_config(FaceRestorer.gfpgan)

        model_path_ = os.path.join(self.model_path, model_config.filename)
        if not os.path.isfile(model_path_):
            download_model(model_path_, model_config.url)

        gfpgan = self._initialize_model(model_path=model_path_)
        return gfpgan

    def _initialize_model(self, model_path: str) -> nn.Module:
        gfpgan = GFPGANv1Clean(
            out_size=512,
            num_style_feat=512,
            channel_multiplier=2,
            decoder_load_path=None,
            fix_decoder=False,
            num_mlp=8,
            input_is_latent=True,
            different_w=True,
            narrow=1,
            sft_half=True
        )

        loadnet = torch.load(model_path)
        keyname = "params_ema" if "params_ema" in loadnet else "params"

        gfpgan.load_state_dict(loadnet[keyname], strict=True)
        gfpgan.eval()
        gfpgan = gfpgan.to(self.get_device())

        return gfpgan

    def get_device(self):
        return settings.device

    # todo: unify with codeformer
    def restore(
            self,
            np_image,
            w: float | None = None,
            enhancement_options=None,
            np_mask: np.ndarray = None
    ):
        def restore_face_executor(cropped_face_t):
            if self.net is None:
                msg = f"{self.name()} is not initialized as expected"
                raise RuntimeError(msg)

            # noinspection PyShadowingNames
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
    def setup_model(cls, dirname: str, face_size: int = 512) -> FaceRestorerGFPGAN:
        return cls(model_path=dirname, face_size=face_size)


# todo: unify with codeformer
def enhance_image(
        image: Image.Image,
        enhancement_options: EnhancementOptions,
        np_mask: np.ndarray = None
) -> Image.Image:
    is_not_current_model = face_restoration_cache.key != FaceRestorer.gfpgan.value
    is_not_cached = not face_restoration_cache.model
    if is_not_cached or is_not_current_model:
        gfpgan = FaceRestorerGFPGAN.setup_model(
            dirname=settings.GFPGAN_MODEL_DIR,
            face_size=enhancement_options.face_enhancement_options.face_size
        )
        face_restoration_cache.model = gfpgan
        face_restoration_cache.key = FaceRestorer.gfpgan.value

    logger.info(
        "Restoring the face with %s (weight: %s)",
        FaceRestorerGFPGAN.name(),
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
