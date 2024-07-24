from typing import Union

import cv2
import numpy as np

from PIL import Image

from . import settings, images, face_analyzer
from .codeformer.codeformer_model import enhance_image
from .conf.settings import EnhancementOptions, DetectionOptions
from .entities.face import FaceArea
from .entities.rect import Rect
from .inferencers.bisenet_mask_generator import BiSeNetMaskGenerator
from .modloader import get_face_swap_model
from .shared import logger, color_generator


# noinspection PyUnusedLocal
def _process_face_image(face: FaceArea, **kwargs) -> Image:
    image = np.array(face.image)
    overlay = image.copy()
    color_iter = color_generator(settings.COLORS)
    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), next(color_iter), -1)
    l, t, r, b = face.face_area_on_image
    cv2.rectangle(overlay, (l, t), (r, b), (0, 0, 0), 10)
    if face.landmarks_on_image is not None:
        for landmark in face.landmarks_on_image:
            cv2.circle(overlay, (int(landmark.x), int(landmark.y)), 6, (0, 0, 0), 10)
    alpha = 0.3
    output = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    return Image.fromarray(output)


def _apply_face_mask(
        swapped_image: np.ndarray,
        target_image: np.ndarray,
        target_face,
        entire_mask_image: np.array
) -> np.ndarray:
    logger.info("Correcting face mask")

    mask_generator = BiSeNetMaskGenerator()
    face = FaceArea(target_image, Rect.from_ndarray(np.array(target_face.bbox)), 1.6, 512, "")
    face_image = np.array(face.image)
    _process_face_image(face)
    face_area_on_image = face.face_area_on_image
    mask = mask_generator.generate_mask(
        face_image,
        face_area_on_image=face_area_on_image,
        affected_areas=["Face"],
        mask_size=0,
        use_minimal_area=True
    )
    mask = cv2.blur(mask, (12, 12))
    # """entire_mask_image = np.zeros_like(target_image)"""
    larger_mask = cv2.resize(mask, dsize=(face.width, face.height))
    entire_mask_image[
        face.top: face.bottom,
        face.left: face.right,
    ] = larger_mask

    result = Image.composite(Image.fromarray(swapped_image), Image.fromarray(target_image),
                             Image.fromarray(entire_mask_image).convert("L"))
    # noinspection PyTypeChecker
    return np.array(result)


def operate(
    *,
    source_img,
    target_img,
    target_img_orig,
    source_faces_index,
    target_faces_index,
    source_faces,
    target_faces,
    gender_source,
    gender_target,
    source_face,
    wrong_gender,
    source_age,
    source_gender,
    swapped,
    enhancement_options,
    detection_options,
):
    from matplotlib import pyplot as plt
    # plt.imshow(np.array(target_img))
    # plt.show()

    result = target_img
    face_swapper = get_face_swap_model()

    for face_num in target_faces_index:
        if source_face is not None and wrong_gender == 0:
            logger.info(f"Detecting target face, index = {str(face_num)}")
            target_face, wrong_gender, target_age, target_gender = face_analyzer.get_face_single(
                img_data=target_img,
                face=target_faces,
                det_thresh=detection_options.det_thresh,
                det_maxnum=detection_options.det_maxnum
            )
            if target_age != "None" or target_gender != "None":
                logger.info(f"Detected target: -{target_age}- y.o. {target_gender}")

            if target_face is not None and wrong_gender == 0:
                logger.info("Swapping source into target")
                swapped_image = face_swapper.get(result, target_face, source_face)
                result = swapped_image
                swapped += 1
            elif wrong_gender == 1:
                raise NotImplemented("Implement wrong gender swapping")

    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    if (enhancement_options is not None and swapped > 0) or enhancement_options.upscale_force:
        result_image = enhance_image(result_image, enhancement_options)

        plt.imshow(result_image)
        plt.show()
    return result_image, swapped


def swap(
    source_image: Image.Image | str,
    target_image: Image.Image | str,
    enhancement_options: Union[EnhancementOptions, None] = None,
    detection_options: Union[DetectionOptions, None] = None,
):
    if detection_options is None:
        detection_options = DetectionOptions()
    if enhancement_options is None:
        enhancement_options = EnhancementOptions()

    source_img = images.get_image(source_image)
    target_img = images.get_image(target_image)
    result_image = target_img

    # noinspection PyTypeChecker
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    target_img_orig = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    entire_mask_image = np.zeros_like(np.array(target_img))

    # TODO: implement for more faces / face swapping
    source_faces_index, target_faces_index = [0], [0]
    if source_faces_index is None:
        source_faces_index = [0]
    if target_faces_index is None:
        target_faces_index = [0]

    # TODO: implement gender selector
    # (0 - No, 1 - Female Only, 2 - Male Only)
    gender_source: int = 0
    gender_target: int = 0

    swapped = 0

    if source_img is not None:
        logger.info('Swapping from source image...')

        source_faces = None
        # noinspection PyTypeChecker
        source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

        logger.info("Analyzing source image...")
        source_faces = face_analyzer.analyze_faces(
            source_img,
            det_thresh=detection_options.det_thresh,
            det_maxnum=detection_options.det_maxnum
        )

        if source_faces is not None:
            logger.info("Analyzing target image...")
            target_faces = face_analyzer.analyze_faces(
                target_img,
                det_thresh=detection_options.det_thresh,
                det_maxnum=detection_options.det_maxnum
            )
            
            if target_faces is None:
                logger.info(f"No faces within target image were found")
                return target_img, 0

            source_face, wrong_gender, source_age, source_gender = face_analyzer.get_face_single(
                img_data=source_img,
                face=source_faces,
                face_index=source_faces_index[0],
                gender_source=gender_source,
                det_thresh=detection_options.det_thresh,
                det_maxnum=detection_options.det_maxnum
            )

            if source_age != "None" or source_gender != "None":
                logger.info(f"Detected source: -{source_age}- y.o. {source_gender}")

            if source_face is not None:
                result_image, swapped = operate(
                    source_img=source_img,
                    target_img=target_img,
                    target_img_orig=target_img_orig,
                    source_faces_index=source_faces_index,
                    target_faces_index=target_faces_index,
                    source_faces=source_faces,
                    target_faces=target_faces,
                    gender_source=gender_source,
                    gender_target=gender_target,
                    source_face=source_face,
                    wrong_gender=wrong_gender,
                    source_age=source_age,
                    source_gender=source_gender,
                    swapped=swapped,
                    enhancement_options=enhancement_options,
                    detection_options=detection_options
                )
        else:
            raise ValueError("No faces within source image were detected")
    else:
        raise NotImplementedError("Any other option is not implemented yet, requires source image")

    return result_image, swapped
