import os
from typing import List, Tuple, Text

import cv2
import numpy as np
import torch
from facexlib.parsing import init_parsing_model
from facexlib.utils.misc import img2tensor
from torchvision.transforms.functional import normalize

from .mask_generator import BaseMaskGenerator
from .mixins import MaskGeneratorMixin, TorchMaskGeneratorMixin
from ..conf.settings import settings
from ..decorators import cpu_offload


class BiSeNetMaskGenerator(BaseMaskGenerator, MaskGeneratorMixin, TorchMaskGeneratorMixin):
    def __init__(self) -> None:
        self.model = init_parsing_model(
            device=settings.device,
            model_rootpath=os.path.join(settings.MODELS_PATH, 'facexlib')
        )

    def name(self):
        return "BiSeNet"

    # noinspection PyMethodOverriding,DuplicatedCode
    @cpu_offload
    def generate_mask(
            self,
            face_image: np.ndarray,
            face_area_on_image: Tuple[int, int, int, int],
            affected_areas: List[Text],
            mask_size: int,
            use_minimal_area: bool,
            fallback_ratio: float = 0.25,
            **kwargs,
    ) -> np.ndarray:
        # --- Convert image to RGB form
        face_image = face_image.copy()
        face_image = face_image[:, :, ::-1]

        if use_minimal_area:
            face_image = self.mask_non_face_areas(face_image, face_area_on_image)

        h, w, _ = face_image.shape
        if w != 512 or h != 512:
            rw = (int(w * (512 / w)) // 8) * 8
            rh = (int(h * (512 / h)) // 8) * 8
            face_image = cv2.resize(face_image, dsize=(rw, rh))

        face_tensor = img2tensor(face_image.astype("float32") / 255.0, float32=True)
        normalize(face_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
        face_tensor = torch.unsqueeze(face_tensor, 0).to(settings.device)

        with torch.no_grad():
            face = self.model(face_tensor)[0]
        face = face.squeeze(0).cpu().numpy().argmax(0)
        face = face.copy().astype(np.uint8)

        mask = self.__to_mask(face, affected_areas)
        mask = self.morph_mask(mask, mask_size=mask_size)

        if w != 512 or h != 512:
            mask = cv2.resize(mask, dsize=(w, h))

        return mask

    @staticmethod
    def __to_mask(face: np.ndarray, affected_areas: List[str]) -> np.ndarray:
        keep_face = "Face" in affected_areas
        keep_neck = "Neck" in affected_areas
        keep_hair = "Hair" in affected_areas
        keep_hat = "Hat" in affected_areas

        mask = np.zeros((face.shape[0], face.shape[1], 3), dtype=np.uint8)
        num_of_class = np.max(face)
        for i in range(1, num_of_class + 1):
            index = np.where(face == i)
            if i < 14 and keep_face:
                mask[index[0], index[1], :] = [255, 255, 255]
            elif i == 14 and keep_neck:
                mask[index[0], index[1], :] = [255, 255, 255]
            elif i == 17 and keep_hair:
                mask[index[0], index[1], :] = [255, 255, 255]
            elif i == 18 and keep_hat:
                mask[index[0], index[1], :] = [255, 255, 255]
        return mask

    @staticmethod
    def mask_non_face_areas(image: np.ndarray, face_area_on_image: Tuple[int, int, int, int]) -> np.ndarray:
        left, top, right, bottom = face_area_on_image
        image = image.copy()
        image[:top, :] = 0
        image[bottom:, :] = 0
        image[:, :left] = 0
        image[:, right:] = 0
        return image

    @staticmethod
    def calculate_mask_coverage(mask: np.ndarray):
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        non_black_pixels = np.count_nonzero(gray_mask)
        total_pixels = gray_mask.size
        return non_black_pixels / total_pixels
