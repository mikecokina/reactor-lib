import os.path
from pathlib import Path
from typing import Union, List, Tuple

import cv2
import numpy as np

from PIL import Image
from PIL import ImageFilter
from insightface.app.common import Face

from . import settings, images, face_analyzer
from .codeformer.codeformer_model import enhance_image
from .conf.settings import EnhancementOptions, DetectionOptions, FaceBlurOptions
from .entities.face import FaceArea
from .entities.rect import Rect
from .inferencers.bisenet_mask_generator import BiSeNetMaskGenerator
from .modloader import get_face_swap_model
from .shared import color_generator, listdir, torch_gc
from .logger import logger


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


def _apply_blur(
    image: np.ndarray,
    target_img: np.ndarray,
    target_face: Face,
    face_blur_options: FaceBlurOptions
) -> Image.Image:

    mask_generator = BiSeNetMaskGenerator()
    face = FaceArea(target_img, Rect.from_ndarray(np.array(target_face.bbox)), 1.6, 512, "")
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
    larger_mask = cv2.resize(mask, dsize=(face.width, face.height))
    entire_mask_image = np.zeros_like(np.array(target_img))
    entire_mask_image[face.top: face.bottom, face.left: face.right] = larger_mask

    image = Image.fromarray(image.astype(np.uint8))
    # Apply Gaussian blur to the entire image
    blurred_image = image.filter(ImageFilter.GaussianBlur(face_blur_options.radius))

    # Adjust the mask intensity based on blur strength
    adjusted_mask = Image.fromarray(entire_mask_image).convert('L').point(lambda p: int(p * face_blur_options.strength))

    # Composite the original and blurred image based on the mask
    return Image.composite(blurred_image, image, adjusted_mask)


def _apply_face_mask(
        swapped_image: np.ndarray,
        target_image: np.ndarray,
        target_face,
        entire_mask_image: np.array
) -> np.ndarray:
    logger.info("Correcting face mask")

    # TODO: use rmbg2.0 as mask generator
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
    entire_mask_image[face.top: face.bottom, face.left: face.right] = larger_mask

    result = Image.composite(
        Image.fromarray(swapped_image), Image.fromarray(target_image),
        Image.fromarray(entire_mask_image).convert("L")
    )
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
        face_mask_correction,
        entire_mask_image,
        face_blur_options
) -> Tuple[Image.Image, int]:
    result = target_img
    face_swapper = get_face_swap_model()
    wrong_gender = 0
    target_face = None

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

                    if face_mask_correction:
                        result = _apply_face_mask(
                            swapped_image=result,
                            target_image=target_img,
                            target_face=target_face,
                            entire_mask_image=entire_mask_image
                        )

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

            if face_mask_correction:
                result = _apply_face_mask(
                    swapped_image=result,
                    target_image=target_img,
                    target_face=target_face,
                    entire_mask_image=entire_mask_image
                )

            swapped += 1

    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    if (enhancement_options is not None and swapped > 0) or enhancement_options.upscale_force:
        result_image = enhance_image(result_image, enhancement_options)

    if face_blur_options.do_face_blur and target_face is not None:
        result_image = _apply_blur(
            image=np.array(result_image),
            target_img=target_img,
            target_face=target_face,
            face_blur_options=face_blur_options
        )

    return result_image, swapped


def _bulk(
        source_image: Image.Image | str,
        input_directory: str,
        output_directory: str,
        source_faces_index: List[int],
        target_faces_index: List[int],
        enhancement_options: EnhancementOptions,
        detection_options: DetectionOptions,
        face_blur_options: FaceBlurOptions,
        face_mask_correction: bool,
        enhance_target_first: bool,
        progressbar: bool
) -> Tuple[None, int]:
    try:
        if progressbar:
            from tqdm import tqdm
        else:
            raise ImportError()
    except ImportError:
        def _tqdm(x, *args, **kwargs):
            return x
        tqdm = _tqdm

    gender_source: int = 0
    gender_target: int = 0
    swapped = 0

    if not os.path.isdir(input_directory):
        raise NotADirectoryError(f"Provided path {input_directory} is not valid directory")

    target_images = listdir(input_directory, filter_ext=settings.SUPPORTED_FORMATS)
    if len(target_images) == 0:
        raise FileNotFoundError(f'No file found supplied directory. '
                                f'Make sure data has valid format. Supported: {str(settings.SUPPORTED_FORMATS)}')

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    # Prepare source faces
    source_img = images.get_image(source_image)
    # noinspection PyTypeChecker
    source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    logger.info("Analyzing source image...")
    source_faces = face_analyzer.analyze_faces(
        source_img,
        det_thresh=detection_options.det_thresh,
        det_maxnum=detection_options.det_maxnum
    )

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

    for target_image in tqdm(target_images):
        logger.info(f"Processing {target_image}")
        target_image = Path(target_image)
        output_image = Path(output_directory) / f"{target_image.stem}.png"
        target_img = images.get_image(str(target_image))

        torch_gc()
        result_image_, swapped_ = _single(
            source_image=source_image,
            target_image=target_img,
            source_faces_index=source_faces_index,
            target_faces_index=target_faces_index,
            enhancement_options=enhancement_options,
            detection_options=detection_options,
            source_face=source_face,
            source_faces=source_faces,
            face_mask_correction=face_mask_correction,
            enhance_target_first=enhance_target_first,
            face_blur_options=face_blur_options
        )

        swapped += swapped_
        logger.info(f"Saving output to {output_image}")
        result_image_.save(output_image, format='PNG')

    return None, swapped


def _single(
        *,
        source_image: Image.Image | str,
        target_image: Image.Image | str,
        source_faces_index: List[int],
        target_faces_index: List[int],
        enhancement_options: EnhancementOptions,
        detection_options: DetectionOptions,
        face_blur_options: FaceBlurOptions,
        source_face=None,
        source_faces: List = None,
        face_mask_correction: bool = False,
        enhance_target_first: bool = False,
) -> Tuple[Image.Image, int]:
    # Single
    source_img = images.get_image(source_image)
    target_img = images.get_image(target_image)
    target_img_org = target_img.copy()

    if enhance_target_first:
        logger.info('Fixing face in target image first')
        target_img = enhance_image(target_img, enhancement_options)

    # noinspection PyTypeChecker
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    target_img_orig = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    entire_mask_image = np.zeros_like(np.array(target_img))

    # TODO: implement gender selector
    # (0 - No, 1 - Female Only, 2 - Male Only)
    gender_source: int = 0
    gender_target: int = 0
    swapped = 0

    if source_img is not None:
        logger.info('Swapping from source image...')

        # noinspection PyTypeChecker
        source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

        if source_faces is None:
            logger.info("Analyzing source image...")
        source_faces = source_faces or face_analyzer.analyze_faces(
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
            return target_img_org, 0

        if source_faces is None:
            logger.warning(f"No faces within source image were found")
            return target_img_org, 0

        # Case when length of source face indices detected is lower than assumed source faces on input.
        if len(source_faces_index) > 1 and (len(source_faces_index) != len(source_faces)):
            raise ValueError(f"Invalid amount of indices within `source_faces`")

        # Case when length of target face indices detected is lower than assumed target faces on input.
        if len(target_faces_index) > 1 and (len(target_faces_index) != len(target_faces)):
            raise ValueError(f"Invalid amount of indices within `target_faces`")

        source_face = source_face or None
        if len(source_faces_index) == 1 and source_face is None:
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
            detection_options=detection_options,
            face_blur_options=face_blur_options,
            face_mask_correction=face_mask_correction,
            entire_mask_image=entire_mask_image,
        )
    else:
        raise NotImplementedError("Any other option is not implemented yet, requires source image")

    return result_image, swapped


def swap(
        source_image: Image.Image | str,
        target_image: Image.Image | str | None = None,
        input_directory: str = None,
        output_directory: str = None,
        source_faces_index: Union[List[int], None] = None,
        target_faces_index: Union[List[int], None] = None,
        enhancement_options: Union[EnhancementOptions, None] = None,
        face_blur_options: Union[FaceBlurOptions, None] = None,
        detection_options: Union[DetectionOptions, None] = None,
        face_mask_correction: bool = False,
        enhance_target_first: bool = False,
        progressbar: bool = False
) -> Tuple[Image.Image, int] | Tuple[None, int]:
    if progressbar:
        logger.get().configure(**dict(SUPPRESS_LOGGER=True))

    if detection_options is None:
        detection_options = DetectionOptions()
    if enhancement_options is None:
        enhancement_options = EnhancementOptions()
    if face_blur_options is None:
        face_blur_options = FaceBlurOptions()

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

    # Batch - single is prioritized
    if target_image is None and input_directory is not None and output_directory is not None:
        return _bulk(
            source_image=source_image,
            input_directory=input_directory,
            output_directory=output_directory,
            source_faces_index=source_faces_index,
            target_faces_index=target_faces_index,
            enhancement_options=enhancement_options,
            detection_options=detection_options,
            face_blur_options=face_blur_options,
            face_mask_correction=face_mask_correction,
            enhance_target_first=enhance_target_first,
            progressbar=progressbar
        )

    # Single - single is prioritized
    else:
        return _single(
            source_image=source_image,
            target_image=target_image,
            source_faces_index=source_faces_index,
            target_faces_index=target_faces_index,
            enhancement_options=enhancement_options,
            face_blur_options=face_blur_options,
            detection_options=detection_options,
            face_mask_correction=face_mask_correction,
            enhance_target_first=enhance_target_first,
        )
