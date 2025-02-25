from typing import Text

import numpy as np
import torch
from insightface.app.common import Face

from . import face_align, image
from .. import settings
from ..conf.settings import FaceSwapper
from .. reswapper.stf_128 import StyleTransferModel


class RESwapper(object):
    def __init__(self, model_path: Text, device: Text = 'cpu'):
        model = StyleTransferModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model.eval()
        self._model = model

        if settings.FACE_SWAPPER == FaceSwapper.reswapper_128:
            self._input_size = (128, 128)
        elif settings.FACE_SWAPPER in [
            FaceSwapper.reswapper_256,
            FaceSwapper.reswapper_256_1567500
        ]:
            self._input_size = (256, 256)
        else:
            raise ValueError("Invalid value for setting.FACE_SWAPPER")

    def _swap_face(
            self,
            target_face_blob: np.ndarray,
            source_face_latent: np.ndarray
    ):
        device = settings.device
        target_tensor = torch.from_numpy(target_face_blob).to(device)
        source_tensor = torch.from_numpy(source_face_latent).to(device)

        with torch.no_grad():
            swapped_tensor = self._model(target_tensor, source_tensor)
        swapped_face = image.postprocess_face(swapped_tensor)
        return swapped_face, swapped_tensor

    def get(
            self,
            result: np.ndarray,
            target_face: Face,
            source_face: Face,
            paste_back=True,
    ) -> np.ndarray:
        target_image = result
        # TODO: use from insightface.utils import face_align

        aligned_target_face, m = face_align.norm_crop2(target_image, target_face.kps, self._input_size[0])
        target_face_blob = image.get_blob(aligned_target_face, self._input_size)
        source_latent = image.get_latent(source_face)

        swapped_face, _ = self._swap_face(target_face_blob, source_latent)
        if paste_back:
            swapped_face = image.blend_swapped_image(swapped_face, target_image, m)

        # np.ndarray BGR form
        return swapped_face
