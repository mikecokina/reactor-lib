from __future__ import annotations

import copy
import os
from functools import cached_property
from typing import Callable, Union, List

import cv2
import numpy as np
import spandrel
import torch

from PIL import Image

from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from .inferencers.maskers import get_face_masker_from_cache, get_hair_masker_from_cache
from .logger import logger
from . import shared, settings, images, face_analyzer
from .entities.face import FaceArea
from .entities.rect import Rect

from .conf.settings import EnhancementOptions
from .shared import bbox_percentage


def _get_mask(
        image: np.ndarray,
        affected_areas: List[str],
        detection_options,
) -> np.ndarray:
    # Affectet areas works for FaceMasker.bisenet only
    try:

        # Select biggest detection
        analyzed_faces = face_analyzer.analyze_faces(
            image,
            det_thresh=detection_options.det_thresh,
            det_maxnum=detection_options.det_maxnum
        )
        covers = [bbox_percentage(*face.bbox, *image.shape[:2]) for face in analyzed_faces]
        recommended_face_index = np.argsort(covers)[-1]
        analyzed_face = analyzed_faces[recommended_face_index]

        if "Hair" in affected_areas:
            mask_generator = get_hair_masker_from_cache().model
        else:
            mask_generator = get_face_masker_from_cache().model

        face = FaceArea(image, Rect.from_ndarray(np.array(analyzed_face.bbox)), 1.6, 512, "")
        face_image = np.array(face.image)
        face_area_on_image = face.face_area_on_image

        face_mask_arr = mask_generator.generate_mask(
            face_image,
            face_area_on_image=face_area_on_image,
            affected_areas=affected_areas,
            mask_size=detection_options.mask_size,
            use_minimal_area=False
        )

        if detection_options.mask_blur_kernel > 0:
            kernel = (detection_options.mask_blur_kernel, detection_options.mask_blur_kernel)
            # noinspection DuplicatedCode
            face_mask_arr = cv2.blur(face_mask_arr, kernel)

        larger_mask = cv2.resize(face_mask_arr, dsize=(face.width, face.height))
        entire_mask_image = np.zeros_like(np.array(image))
        entire_mask_image[face.top: face.bottom, face.left: face.right] = larger_mask

    except IndexError:
        # In case of error, mask is an entire image
        entire_mask_image = np.ones(image.shape, dtype=np.uint8) * 255

    return entire_mask_image


def get_hair_mask(
        image: np.ndarray,
        detection_options,
):
    return _get_mask(image, affected_areas=["Hair"], detection_options=detection_options)


def get_face_mask(
        image: np.ndarray,
        detection_options,
) -> np.ndarray:
    return _get_mask(image, affected_areas=["Face"], detection_options=detection_options)


# noinspection DuplicatedCode
def restore_with_face_helper(
        np_image: np.ndarray,
        face_helper: FaceRestoreHelper,
        restore_face: Callable[[torch.Tensor], torch.Tensor],
        enhancement_options: EnhancementOptions,
        np_mask: Union[np.ndarray, None]
) -> np.ndarray:
    from torchvision.transforms.functional import normalize
    np_image = np_image[:, :, ::-1]
    original_resolution = np_image.shape[0:2]

    face_enhancement_options = enhancement_options.face_enhancement_options
    face_detection_options = face_enhancement_options.face_detection_options
    hair_detection_options = face_enhancement_options.hair_detection_options
    restore_face_only = enhancement_options.face_enhancement_options.restore_face_only
    paste_back_hair = enhancement_options.face_enhancement_options.paste_back_hair

    try:
        # 1) Enhance faces.
        logger.debug("Detecting faces...")
        restored_faces = []

        # Restoration image face helper
        face_helper.clean_all()
        face_helper.read_image(np_image)
        face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        face_helper.align_warp_face()

        logger.debug(f"Found {str(len(face_helper.cropped_faces))} faces, restoring...")

        for cropped_face in face_helper.cropped_faces:
            cropped_face_t = images.bgr_image_to_rgb_tensor(cropped_face / 255.0)
            normalize(cropped_face_t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(settings.device)

            # noinspection PyBroadException
            try:
                with torch.no_grad():
                    cropped_face_t = restore_face(cropped_face_t)
                shared.torch_gc()
            except Exception as e:
                logger.error(f'Failed face-restoration inference, {str(e)}')

            restored_face = images.rgb_tensor_to_bgr_image(cropped_face_t, min_max=(-1, 1))
            restored_face = (restored_face * 255.0).astype('uint8')

            restored_faces.append(restored_face)

        # Prepare face helper for further operations
        if np_mask is not None:
            np_mask = np_mask[:, :, ::-1]
            # Deep copy face helper for mask use
            face_helper_mask = copy.deepcopy(face_helper)
            face_helper_mask.read_image(np_mask)

            # Reset cropped_faces fromm copied object
            face_helper_mask.cropped_faces = []
            face_helper_mask.align_warp_face()
        else:
            face_helper_mask = face_helper

        # 2) Replace face only.
        if restore_face_only:
            logger.debug("Proessing face")

            for face_index in range(0, len(restored_faces)):
                restored_face = restored_faces[face_index]
                cropped_mask_source = face_helper_mask.cropped_faces[face_index]

                # Find face mask
                face_mask_arr = get_face_mask(
                    image=cropped_mask_source,
                    detection_options=face_detection_options
                )
                # Alpha blend restored image with source face image
                restored_face = np.array(Image.composite(
                    Image.fromarray(restored_face),
                    Image.fromarray(cropped_mask_source),
                    Image.fromarray(face_mask_arr).convert('L')
                ))

                restored_faces[face_index] = restored_face

        # 3) Replace hair from original image.
        if paste_back_hair:
            logger.debug("Proessing hair")

            for face_index in range(0, len(restored_faces)):
                restored_face = restored_faces[face_index]
                cropped_mask_source = face_helper_mask.cropped_faces[face_index]

                # Find face mask
                hair_mask_arr = get_hair_mask(
                    image=cropped_mask_source,
                    detection_options=hair_detection_options
                )
                blended_face = np.array(Image.composite(
                    Image.fromarray(cropped_mask_source),
                    Image.fromarray(restored_face),
                    Image.fromarray(hair_mask_arr).convert('L')
                ))
                restored_faces[face_index] = blended_face

        # Put restored face back to original image
        logger.debug("Merging restored faces into image")

        for restored_face in restored_faces:
            # Add to helper for inverse affine transformation
            face_helper.add_restored_face(restored_face)

        face_helper.get_inverse_affine(None)
        result_img = face_helper.paste_faces_to_input_image()
        result_img = result_img[:, :, ::-1]
        if original_resolution != result_img.shape[0:2]:
            result_img = cv2.resize(
                result_img,
                (0, 0),
                fx=original_resolution[1] / result_img.shape[1],
                fy=original_resolution[0] / result_img.shape[0],
                interpolation=cv2.INTER_LINEAR,
            )
        logger.debug("Face restoration complete")
    finally:
        face_helper.clean_all()
    return result_img


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

    def __init__(self, model_path: str, face_size: int = 512):
        super().__init__()
        self.net: Union[spandrel.ModelDescriptor, None] = None
        self.model_path = model_path
        self.face_size = face_size
        os.makedirs(model_path, exist_ok=True)

    # noinspection PyMethodMayBeStatic
    def create_face_helper(self, device) -> FaceRestoreHelper:
        from facexlib.detection import retinaface
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        if hasattr(retinaface, 'device'):
            retinaface.device = device
        # update face_size=512 to self.face_size when find out how to run
        # codeformer with 1024 input or retrain or use different model
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
            enhancement_options,
            np_mask: np.ndarray = None
    ) -> np.ndarray:
        self.net = self.load_net()

        self.send_model_to(self.get_device())
        return restore_with_face_helper(
            np_image,
            self.face_helper,
            restore_face,
            enhancement_options,
            np_mask
        )
