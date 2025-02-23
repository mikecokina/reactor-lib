import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from facexlib.detection import retinaface
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from tqdm import tqdm

from reactorlib import settings, DetectionOptions
from reactorlib.restoration import get_face_mask
from reactorlib.shared import listdir


def create_face_helper(device: str = "cuda"):
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


def face_mask(
        image: Image.Image,
        face_helper: FaceRestoreHelper,
        face_detection_options: DetectionOptions
) -> Tuple[Image.Image, Image.Image]:
    np_image = np.array(image, dtype=np.uint8)[:, :, ::-1]
    original_resolution = np_image.shape[0:2]

    face_helper.clean_all()
    face_helper.read_image(np_image)
    face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
    face_helper.align_warp_face()

    for cropped_face in face_helper.cropped_faces:
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

    result_img = Image.fromarray(np_image[:, :, ::-1])
    result_img.putalpha(Image.fromarray(result_mask).convert('L'))

    return result_img, Image.fromarray(result_mask).convert('L')


def main():
    input_dir = ""
    composition_dir = ""
    mask_dir = ""

    paths = listdir(input_dir)
    options = DetectionOptions(
        det_thresh=0.25,
        det_maxnum=0,
        reverse_detection_order=False
    )
    f_helper = create_face_helper()
    os.makedirs(composition_dir, exist_ok=True)

    for path in tqdm(paths):
        image = Image.open(path).convert('RGB')
        image, mask = face_mask(image, f_helper, options)
        output_path = Path(composition_dir) / Path(path).name
        mask_path = Path(mask_dir) / Path(path).name

        image.save(output_path)
        mask.save(mask_path)


if __name__ == '__main__':
    main()
