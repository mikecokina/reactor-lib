from __future__ import annotations

import os
from functools import cached_property
from typing import Callable, Union

import cv2
import numpy as np
import spandrel
import torch


from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from . import shared, settings, images
from .logger import logger


def restore_with_face_helper(
        np_image: np.ndarray,
        face_helper: FaceRestoreHelper,
        restore_face: Callable[[torch.Tensor], torch.Tensor],
) -> np.ndarray:
    from torchvision.transforms.functional import normalize
    np_image = np_image[:, :, ::-1]
    original_resolution = np_image.shape[0:2]

    try:
        logger.debug("Detecting faces...")

        face_helper.clean_all()
        face_helper.read_image(np_image)
        face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        face_helper.align_warp_face()

        logger.debug(f"Found {str(len(face_helper.cropped_faces))} faces, restoring")

        for cropped_face in face_helper.cropped_faces:
            cropped_face_t = images.bgr_image_to_rgb_tensor(cropped_face / 255.0)
            normalize(cropped_face_t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(settings.device)

            # noinspection PyBroadException
            try:
                with torch.no_grad():
                    cropped_face_t = restore_face(cropped_face_t)
                shared.torch_gc()
            except Exception:
                print('Failed face-restoration inference')

            restored_face = images.rgb_tensor_to_bgr_image(cropped_face_t, min_max=(-1, 1))
            restored_face = (restored_face * 255.0).astype('uint8')
            face_helper.add_restored_face(restored_face)

        logger.debug("Merging restored faces into image")
        face_helper.get_inverse_affine(None)
        img = face_helper.paste_faces_to_input_image()
        img = img[:, :, ::-1]
        if original_resolution != img.shape[0:2]:
            img = cv2.resize(
                img,
                (0, 0),
                fx=original_resolution[1] / img.shape[1],
                fy=original_resolution[0] / img.shape[0],
                interpolation=cv2.INTER_LINEAR,
            )
        logger.debug("Face restoration complete")
    finally:
        face_helper.clean_all()
    return img


class FaceRestoration(object):
    # noinspection PyMethodMayBeStatic
    def name(self):
        return "None"

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def restore(self, np_image, **kwargs):
        return np_image


class CommonFaceRestoration(FaceRestoration):
    net: torch.Module | None
    model_url: str
    model_download_name: str

    def __init__(self, model_path: str):
        super().__init__()
        self.net: Union[spandrel.ModelDescriptor, None] = None
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)

    @staticmethod
    def create_face_helper(device) -> FaceRestoreHelper:
        from facexlib.detection import retinaface
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        if hasattr(retinaface, 'device'):
            retinaface.device = device
        return FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=device,
            model_rootpath=os.path.join(settings.MODELS_PATH, 'facexlib')
        )

    @cached_property
    def face_helper(self) -> FaceRestoreHelper:
        return self.create_face_helper(self.get_device())

    def send_model_to(self, device):
        if self.net:
            logger.debug(f"Sending {str(self.net)} to {str(device)}")
            self.net.to(device)
        if self.face_helper:
            logger.debug(f"Sending face helper to {str(device)}")
            self.face_helper.face_det.to(device)
            self.face_helper.face_parse.to(device)

    def get_device(self):
        raise NotImplementedError("get_device must be implemented by subclasses")

    def load_net(self) -> torch.Module:
        raise NotImplementedError("load_net must be implemented by subclasses")

    def restore_with_helper(
        self,
        np_image: np.ndarray,
        restore_face: Callable[[torch.Tensor], torch.Tensor],
    ) -> np.ndarray:
        self.net = self.load_net()

        # try:
        self.send_model_to(self.get_device())
        return restore_with_face_helper(np_image, self.face_helper, restore_face)
        # finally:
        #     if shared.opts.face_restoration_unload:
        #         self.send_model_to(devices.cpu)
