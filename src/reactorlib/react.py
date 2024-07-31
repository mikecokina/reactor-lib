from typing import Union, List

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
    source_img: np.ndarray,
    target_img: np.ndarray,
    target_img_orig: np.ndarray,
    source_faces_index: List[int],
    target_faces_index: List[int],
    source_faces,
    target_faces,
    source_face,
    gender_source,
    gender_target,
    swapped,
    enhancement_options,
    detection_options,
):
    result = target_img
    face_swapper = get_face_swap_model()
    wrong_gender = 0

    # Single source and several targets
    if len(source_faces_index) == 1:
        for face_num in target_faces_index:
            # TODO: sanitize index error (face_num === max(target_faces_index) > len(target_faces))
            if source_face is not None and wrong_gender == 0:
                logger.info(f"Analyzing target face, index = {str(face_num)}")
                target_face, wrong_gender, target_age, target_gender = face_analyzer.get_face_single(
                    img_data=target_img,
                    face=target_faces,
                    face_index=face_num,
                    gender_target=gender_target,
                    det_thresh=detection_options.det_thresh,
                    det_maxnum=detection_options.det_maxnum
                )
                if target_age != "None" or target_gender != "None":
                    logger.info(f"Analyzed target: -{target_age}- y.o. {target_gender}")

                if target_face is not None and wrong_gender == 0:
                    logger.info("Swapping source into target")
                    result = face_swapper.get(result, target_face, source_face)
                    swapped += 1
                elif wrong_gender == 1:
                    raise NotImplemented("Implement wrong gender swapping")

    # Several sources and several targets defined via index mapping
    elif len(source_faces_index) == len(target_faces_index):
        for source_face_num, targe_face_num in zip(source_faces_index, target_faces_index):

            logger.info(f"Analyzing source face, index = {str(source_face_num)}")
            source_face, wrong_gender, source_age, source_gender = face_analyzer.get_face_single(
                img_data=source_img,
                face=source_faces,
                face_index=source_face_num,
                gender_source=gender_source,
                det_thresh=detection_options.det_thresh,
                det_maxnum=detection_options.det_maxnum
            )
            if source_age != "None" or source_gender != "None":
                logger.info(f"Analyzed source: -{source_age}- y.o. {source_gender}")

            if source_face is None:
                logger.error(f"Cannot extract source face, index = {str(source_face_num)}")
                continue

            logger.info(f"Analyzing target face, index = {str(targe_face_num)}")
            target_face, wrong_gender, target_age, target_gender = face_analyzer.get_face_single(
                img_data=target_img,
                face=target_faces,
                face_index=targe_face_num,
                gender_target=gender_target,
                det_thresh=detection_options.det_thresh,
                det_maxnum=detection_options.det_maxnum
            )
            if target_age != "None" or target_gender != "None":
                logger.info(f"Analyzed target: -{target_age}- y.o. {target_gender}")

            if target_face is None:
                logger.error(f"Cannot extract target face, index = {str(targe_face_num)}")
                continue

            logger.info("Swapping source into target")
            result = face_swapper.get(result, target_face, source_face)
            swapped += 1

    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    if (enhancement_options is not None and swapped > 0) or enhancement_options.upscale_force:
        result_image = enhance_image(result_image, enhancement_options)
    return result_image, swapped


def swap(
    source_image: Image.Image | str,
    target_image: Image.Image | str,
    source_faces_index: Union[List[int], None] = None,
    target_faces_index: Union[List[int], None] = None,
    enhancement_options: Union[EnhancementOptions, None] = None,
    detection_options: Union[DetectionOptions, None] = None,
):
    if detection_options is None:
        detection_options = DetectionOptions()
    if enhancement_options is None:
        enhancement_options = EnhancementOptions()

    source_img = images.get_image(source_image)
    target_img = images.get_image(target_image)
    # result_image = target_img

    # noinspection PyTypeChecker
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    target_img_orig = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    entire_mask_image = np.zeros_like(np.array(target_img))

    if source_faces_index is None:
        source_faces_index = [0]
    if target_faces_index is None:
        target_faces_index = [0]

    if len(source_faces_index) != 0 \
            and len(source_faces_index) != 1 \
            and len(source_faces_index) != len(target_faces_index):
        raise ValueError(
            "Source faces must have no entries (default=0), one entry, or same number of entries as target faces."
        )

    # TODO: implement gender selector
    # (0 - No, 1 - Female Only, 2 - Male Only)
    gender_source: int = 0
    gender_target: int = 0

    swapped = 0

    if source_img is not None:
        logger.info('Swapping from source image...')

        # noinspection PyTypeChecker
        source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

        logger.info("Analyzing source image...")
        source_faces = face_analyzer.analyze_faces(
            source_img,
            det_thresh=detection_options.det_thresh,
            det_maxnum=detection_options.det_maxnum
        )

        logger.info("Analyzing target image...")
        target_faces = face_analyzer.analyze_faces(
            target_img,
            det_thresh=detection_options.det_thresh,
            det_maxnum=detection_options.det_maxnum
        )

        # Return original target image if no faces were detected
        if target_faces is None:
            logger.warning(f"No faces within target image were found")
            return target_img, 0

        if source_faces is None:
            logger.warning(f"No faces within source image were found")
            return target_img, 0

        # Case when length of source face indices detected is lower than assumed source faces on input.
        if len(source_faces_index) > 1 and (len(source_faces_index) != len(source_faces)):
            raise ValueError(f"Invalid amount of indices within `source_faces`")

        # Case when length of target face indices detected is lower than assumed target faces on input.
        if len(target_faces_index) > 1 and (len(target_faces_index) != len(target_faces)):
            raise ValueError(f"Invalid amount of indices within `target_faces`")

        source_face = None
        if len(source_faces_index) == 1:
            logger.info(f"Analyzing source face, index = {str(source_faces_index[0])}")
            source_face, wrong_gender, source_age, source_gender = face_analyzer.get_face_single(
                img_data=source_img,
                face=source_faces,
                face_index=source_faces_index[0],
                gender_source=gender_source,
                det_thresh=detection_options.det_thresh,
                det_maxnum=detection_options.det_maxnum
            )
            if source_age != "None" or source_gender != "None":
                logger.info(f"Analyzed source: -{source_age}- y.o. {source_gender}")

        result_image, swapped = operate(
            source_img=source_img,
            target_img=target_img,
            target_img_orig=target_img_orig,
            source_faces_index=source_faces_index,
            target_faces_index=target_faces_index,
            source_faces=source_faces,
            source_face=source_face,
            target_faces=target_faces,
            gender_source=gender_source,
            gender_target=gender_target,
            swapped=swapped,
            enhancement_options=enhancement_options,
            detection_options=detection_options
        )
    else:
        raise NotImplementedError("Any other option is not implemented yet, requires source image")

    return result_image, swapped