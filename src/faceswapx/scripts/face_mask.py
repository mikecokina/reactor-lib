from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from faceswapx import DetectionOptions
from faceswapx.restoration import get_face_mask


def mask_frame(
        image: Image.Image,
        face_helper: FaceRestoreHelper,
        face_detection_options: DetectionOptions,
        face_index: int = 0
) -> Tuple[Image.Image, Image.Image]:
    image = image.convert('RGB')
    np_image = np.array(image, dtype=np.uint8)[:, :, ::-1]
    original_resolution = np_image.shape[0:2]

    face_helper.clean_all()
    face_helper.read_image(np_image)
    face_helper.get_face_landmarks_5(
        only_center_face=False,
        resize=640,
        eye_dist_threshold=5
    )
    face_helper.align_warp_face()

    if face_helper.cropped_faces is not None and len(face_helper.cropped_faces) <= face_index + 1:
        cropped_face = face_helper.cropped_faces[face_index]
        face_mask_arr = get_face_mask(
            image=cropped_face,
            detection_options=face_detection_options
        )
        face_helper.add_restored_face(face_mask_arr)

        face_helper.get_inverse_affine(None)
        face_helper.read_image(np.zeros((*original_resolution, 3), dtype=np.uint8))

        face_helper.use_parse = False
        result_mask = face_helper.paste_faces_to_input_image()
        result_mask = result_mask[:, :, ::-1]

        if original_resolution != result_mask.shape[0:2]:
            result_mask = cv2.resize(
                result_mask,
                (0, 0),
                fx=original_resolution[1] / result_mask.shape[1],
                fy=original_resolution[0] / result_mask.shape[0],
                interpolation=cv2.INTER_LINEAR,
            )
    else:
        result_mask = np.zeros((*original_resolution, 3), dtype=np.uint8)

    result_img = Image.fromarray(np_image[:, :, ::-1])
    result_img.putalpha(Image.fromarray(result_mask).convert('L'))

    return result_img, Image.fromarray(result_mask).convert('L')
