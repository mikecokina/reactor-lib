import os
from typing import Union

import cv2
import numpy as np
import onnxruntime

from PIL import Image
from onnxruntime import InferenceSession
from torchvision import transforms

from .mixins import MaskGeneratorMixin
from .. conf.settings import settings, FaceMasker, FaceMaskModels
from .. shared import download_model


class BiRefNetMaskGenerator(MaskGeneratorMixin):
    def __init__(self) -> None:
        self._image_size = 512

        face_masker_options = FaceMaskModels.get_config(FaceMasker.birefnet_large)
        self.model_path = os.path.join(settings.BIREFNET_MODEL_DIR, face_masker_options.filename)

        if not os.path.exists(self.model_path):
            download_model(self.model_path, face_masker_options.url)

        self.mask_model: InferenceSession = self._initialize_model()

    @staticmethod
    def check_state_dict(state_dict, unwanted_prefixes=None):
        if unwanted_prefixes is not None:
            unwanted_prefixes = ['_orig_mod.', 'module.']

        for k, v in list(state_dict.items()):
            for unwanted_prefix in unwanted_prefixes:
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                    break
        return state_dict

    def _initialize_model(self):
        onnx_session = onnxruntime.InferenceSession(self.model_path, providers=settings.PROVIDERS)
        return onnx_session

    @staticmethod
    def name():
        return "BiRefNet"

    # noinspection DuplicatedCode,PyUnusedLocal
    def generate_mask(
            self,
            face_image: np.ndarray,
            mask_size: int,
            **kwargs,
    ):
        # Comes as cv2 BGR iamge
        face_image = face_image[:, :, ::-1]

        h, w, _ = face_image.shape
        if w != 512 or h != 512:
            rw = (int(w * (512 / w)) // 8) * 8
            rh = (int(h * (512 / h)) // 8) * 8
            face_image = cv2.resize(face_image, dsize=(rw, rh))

        mask = self._get_mask(face_image)
        mask = self.morph_mask(mask, mask_size=mask_size)

        if w != 512 or h != 512:
            mask = cv2.resize(mask, dsize=(w, h))

        mask = np.array(Image.fromarray(mask.astype(np.uint8)).convert('RGB'))
        return mask

    def _get_mask(
            self,
            image: Union[np.ndarray, Image.Image],
    ):

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        image = image.convert('RGB')

        transform_image = transforms.Compose([
            transforms.Resize((self._image_size, self._image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        input_image = transform_image(image).unsqueeze(0)
        dtype = np.float32 if settings.NO_HALF else np.float16
        input_image_numpy = np.array(input_image, dtype=dtype)

        input_name = self.mask_model.get_inputs()[0].name
        onnx_output = self.mask_model.run(
            None,
            {input_name: input_image_numpy}
        )[-1]

        # Process the output
        pred = np.squeeze(onnx_output)
        pred = 1 / (1 + np.exp(-pred))  # Apply sigmoid activation
        pred = np.clip(pred, 0, 1)  # Clamp values to [0, 1]

        # Convert to uint8 for visualization
        pred = (pred * 255).astype(np.uint8)

        # Convert to PIL Image and resize to original size
        pred_pil = Image.fromarray(pred).convert('L')
        mask = pred_pil.resize(image.size, Image.Resampling.BILINEAR)

        # Return the mask as a NumPy array
        return np.array(mask).astype(np.uint8)
