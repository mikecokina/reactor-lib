import os.path
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Union, List, Tuple

import cv2
import numpy as np

from PIL import Image
from PIL import ImageFilter
from insightface.app.common import Face

from . import settings, images, face_analyzer
from .codeformer.codeformer_model import enhance_image
from .conf.settings import EnhancementOptions, DetectionOptions, FaceBlurOptions, FaceSwapper
from .entities.face import FaceArea
from .entities.rect import Rect
from .inferencers.maskers import get_masker_cache
from .modloader import get_inswapper_model, get_reswapper_model
from .nudity import is_nsfw
from .shared import color_generator, listdir, torch_gc, get_tqdm_cls
from .logger import logger
from .video.utils import video2frames, frames2video


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


def pad_bbox(bbox: Union[Tuple, List], w: int, h: int, pad_percent: float):
    if pad_percent == 0:
        return bbox

    x1, y1, x2, y2 = bbox

    # Calculate padding in pixels
    pad_x = (x2 - x1) * pad_percent / 100
    pad_y = (y2 - y1) * pad_percent / 100

    # Apply padding
    padded_x1 = max(0, x1 - pad_x)  # Clamp to minimum 0
    padded_y1 = max(0, y1 - pad_y)  # Clamp to minimum 0
    padded_x2 = min(w, x2 + pad_x)  # Clamp to maximum width
    padded_y2 = min(h, y2 + pad_y)  # Clamp to maximum height

    return int(padded_x1), int(padded_y1), int(padded_x2), int(padded_y2)


# noinspection DuplicatedCode
def _apply_blur(
        image: np.ndarray,
        target_img: np.ndarray,
        target_face: Face,
        face_blur_options: FaceBlurOptions
) -> Image.Image:
    random.seed(face_blur_options.seed)

    mask_generator = get_masker_cache().model
    face = FaceArea(target_img, Rect.from_ndarray(np.array(target_face.bbox)), 1.6, 512, "")
    face_image = np.array(face.image)
    _process_face_image(face)
    face_area_on_image = face.face_area_on_image
    face_mask_arr = mask_generator.generate_mask(
        face_image,
        face_area_on_image=face_area_on_image,
        affected_areas=["Face", "Neck", "Hair"],
        mask_size=face_blur_options.mask_size,
        use_minimal_area=False
    )
    face_mask_arr = cv2.blur(face_mask_arr, (12, 12))

    larger_mask = cv2.resize(face_mask_arr, dsize=(face.width, face.height))
    entire_mask_image = np.zeros_like(np.array(target_img))
    entire_mask_image[face.top: face.bottom, face.left: face.right] = larger_mask
    # entire_mask_image_pil = Image.fromarray(entire_mask_image).convert('L')

    # Transform image to PIL
    pil_image = Image.fromarray(image.astype(np.uint8))
    image_ = pil_image.copy()

    # Apply Gaussian blur
    if face_blur_options.do_face_blur:
        image_ = image_.filter(ImageFilter.GaussianBlur(face_blur_options.blur_radius))

    # Apply Video Noise Effect
    if face_blur_options.do_video_noise:
        # Get original image size
        original_size = image_.size
        # Step 1: Pixelate the image by resizing it down and back up
        small_size = (original_size[0] // face_blur_options.noise_pixel_size, original_size[1]
                      // face_blur_options.noise_pixel_size)
        # Resizing down using nearest neighbor (blocky effect)
        pixelated = image_.resize(small_size, Image.Resampling.NEAREST)
        pixelated = pixelated.resize(original_size, Image.Resampling.NEAREST)  # Resize back to original size
        # Step 2: Add random noise (grainy effect) on top of the pixelated image
        noisy_image = Image.new("RGB", original_size)
        for x in range(original_size[0]):
            for y in range(original_size[1]):
                # Get pixel from the pixelated image
                pixel = pixelated.getpixel((x, y))

                # Add noise to each channel (R, G, B)
                noisy_pixel = tuple(
                    min(255, max(0, int(pixel[i] + random.uniform(-1, 1) * face_blur_options.noise_strength * 255)))
                    for i in range(3)
                )
                noisy_image.putpixel((x, y), noisy_pixel)
        image_ = noisy_image

    adjusted_mask = Image.fromarray(entire_mask_image).convert('L').point(
        lambda p: int(p * face_blur_options.blur_strength)
    )

    return Image.composite(image_, pil_image, adjusted_mask)


def _apply_face_mask(
        swapped_image: np.ndarray,
        target_image: np.ndarray,
        target_face,
        entire_mask_image: np.array,
        face_mask_correction_size: int = 0
) -> np.ndarray:
    logger.info("Correcting face mask")

    mask_generator = get_masker_cache().model

    face = FaceArea(target_image, Rect.from_ndarray(np.array(target_face.bbox)), 1.6, 512, "")
    face_image = np.array(face.image)
    _process_face_image(face)
    face_area_on_image = face.face_area_on_image
    mask = mask_generator.generate_mask(
        face_image,
        face_area_on_image=face_area_on_image,
        affected_areas=["Face"],
        mask_size=face_mask_correction_size,
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


class FaceSwapperCache(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceSwapperCache, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_model"):
            self._model = None  # Initialize instance attribute
        if not hasattr(self, "_key"):
            self._key = None

    @property
    def model(self):
        """Getter for the model attribute"""
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def key(self):
        """Getter for the model attribute"""
        return self._key

    @key.setter
    def key(self, value):
        self._key = value


face_swapper_cache = FaceSwapperCache()


def get_face_swapper_cache() -> FaceSwapperCache:
    if (face_swapper_cache.model is None) or (face_swapper_cache.key != settings.FACE_SWAPPER.value):
        if settings.FACE_SWAPPER == FaceSwapper.inswapper:
            face_swapper_cache.model = get_inswapper_model()
        elif settings.FACE_SWAPPER in [
            FaceSwapper.reswapper_128,
            FaceSwapper.reswapper_256,
            FaceSwapper.reswapper_256_1567500
        ]:
            face_swapper_cache.model = get_reswapper_model()
        else:
            raise NotImplementedError(f"{settings.FACE_SWAPPER} not implemented")
        face_swapper_cache.key = settings.FACE_SWAPPER.value
    return face_swapper_cache


# noinspection DuplicatedCode
def operate(
        *,
        source_img: np.ndarray,
        target_img: np.ndarray,
        source_faces_index: List[int],
        target_faces_index: List[int],
        source_faces,
        target_faces,
        source_face,
        swapped,
        enhancement_options,
        face_mask_correction,
        entire_mask_image,
        face_blur_options,
        face_mask_correction_size: int,
        reverse_detection_order: bool
) -> Tuple[Image.Image, int]:
    result = target_img
    face_swapper = get_face_swapper_cache().model

    wrong_gender = 0
    target_face = None

    # NSFW test
    nsfw_detected = is_nsfw(target_img)
    if nsfw_detected:
        # Return black image if NSFW content detected.
        return Image.fromarray(target_img.copy() * 0), 0

    # Single source and several targets
    if len(source_faces_index) == 1:
        for face_num in target_faces_index:
            # TODO: sanitize index error (face_num === max(target_faces_index) > len(target_faces))
            if source_face is not None and wrong_gender == 0:
                logger.info(f"Analyzing target face, index = {str(face_num)}")
                target_face, wrong_gender, target_age, target_gender = face_analyzer.get_face_single(
                    faces=target_faces,
                    face_index=face_num,
                    reverse_detection_order=reverse_detection_order
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
                            entire_mask_image=entire_mask_image,
                            face_mask_correction_size=face_mask_correction_size
                        )

                    swapped += 1
                elif wrong_gender == 1:
                    raise NotImplemented("Implement wrong gender swapping")

    # Several sources and several targets defined via index mapping
    elif len(source_faces_index) == len(target_faces_index):
        for source_face_num, targe_face_num in zip(source_faces_index, target_faces_index):

            logger.info(f"Analyzing source face, index = {str(source_face_num)}")
            source_face, wrong_gender, source_age, source_gender = face_analyzer.get_face_single(
                faces=source_faces,
                face_index=source_face_num,
            )
            if source_age != "None" or source_gender != "None":
                logger.info(f"Analyzed source: -{source_age}- y.o. {source_gender}")

            if source_face is None:
                logger.error(f"Cannot extract source face, index = {str(source_face_num)}")
                continue

            logger.info(f"Analyzing target face, index = {str(targe_face_num)}")
            target_face, wrong_gender, target_age, target_gender = face_analyzer.get_face_single(
                faces=target_faces,
                face_index=targe_face_num,
                reverse_detection_order=reverse_detection_order
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
                    entire_mask_image=entire_mask_image,
                    face_mask_correction_size=face_mask_correction_size
                )

            swapped += 1

    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    if enhancement_options.face_enhancement_options.do_enhancement and swapped > 0:
        result_image = enhance_image(result_image, enhancement_options)

    if (face_blur_options.do_face_blur or face_blur_options.do_video_noise) and target_face is not None:
        logger.info("Applying blur to final image")
        # noinspection PyTypeChecker
        result_image = _apply_blur(
            image=np.array(result_image),
            target_img=target_img,
            target_face=target_face,
            face_blur_options=face_blur_options
        )

    return result_image, swapped


# noinspection DuplicatedCode
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
        face_mask_correction_size: int,
        skip_if_exists: bool,
        progressbar: bool,
        **kwargs
) -> Tuple[None, int]:
    tqdm = get_tqdm_cls(progressbar=progressbar)
    tqdm = kwargs.get("gr_progressbar", tqdm)

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
            faces=source_faces,
            face_index=source_faces_index[0],
        )
        if source_age != "None" or source_gender != "None":
            logger.info(f"Analyzed source: -{source_age}- y.o. {source_gender}")

    for target_image in tqdm(target_images, desc="Swapping", unit="images"):
        logger.info(f"Processing {target_image}")
        target_image = Path(target_image)
        output_image = Path(output_directory) / f"{target_image.stem}.png"

        if skip_if_exists and os.path.isfile(output_image):
            logger.info(f"Image {output_image} already exists")
            continue

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
            face_blur_options=face_blur_options,
            face_mask_correction_size=face_mask_correction_size
        )

        swapped += swapped_
        logger.info(f"Saving output to {output_image}")
        result_image_.save(output_image, format='PNG')

    return None, swapped


# noinspection DuplicatedCode
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
        face_mask_correction_size: int,
        face_mask_correction: bool = False,
) -> Tuple[Image.Image, int]:
    # Single
    source_img = images.get_image(source_image)
    target_img = images.get_image(target_image)
    target_img_org = target_img.copy()

    if enhancement_options.face_enhancement_options.enhance_target:
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
                faces=source_faces,
                face_index=source_faces_index[0],
            )
            if source_age != "None" or source_gender != "None":
                logger.info(f"Analyzed source: -{source_age}- y.o. {source_gender}")

        result_image, swapped = operate(
            source_img=source_img,
            target_img=target_img,
            source_faces_index=source_faces_index,
            target_faces_index=target_faces_index,
            source_faces=source_faces,
            source_face=source_face,
            target_faces=target_faces,
            swapped=swapped,
            enhancement_options=enhancement_options,
            face_blur_options=face_blur_options,
            face_mask_correction=face_mask_correction,
            entire_mask_image=entire_mask_image,
            face_mask_correction_size=face_mask_correction_size,
            reverse_detection_order=detection_options.reverse_detection_order
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
        face_mask_correction_size: int = 0,
        skip_if_exists: bool = False,
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
            face_mask_correction_size=face_mask_correction_size,
            skip_if_exists=skip_if_exists,
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
            face_mask_correction_size=face_mask_correction_size
        )


image_swap = swap


def video_swap(
        source_image: Union[Image.Image, str],
        target_video: Union[Path, str],
        output_directory: Union[Path, str] = None,
        source_faces_index: Union[List[int], None] = None,
        target_faces_index: Union[List[int], None] = None,
        enhancement_options: Union[EnhancementOptions, None] = None,
        face_blur_options: Union[FaceBlurOptions, None] = None,
        detection_options: Union[DetectionOptions, None] = None,
        face_mask_correction: bool = False,
        face_mask_correction_size: int = 0,
        high_quality: bool = False,
        progressbar: bool = False,
        keep_frames: bool = False,
        desired_fps: float = 25.0
):
    if progressbar:
        logger.get().configure(**dict(SUPPRESS_LOGGER=True))

    suffix_ = int(time.time())
    tmp_in_frames_dir = str(Path(tempfile.gettempdir()) / f"frames_in_{suffix_}")
    output_frames_dir = str(Path(output_directory) / f"frames_out_{suffix_}")
    output_video_file = Path(output_directory) / f"{Path(target_video).stem}_swapped.mp4"
    os.makedirs(output_directory, exist_ok=True)

    try:
        # Split video to frames
        _, effective_fps = video2frames(
            video_path=target_video,
            output_directory=tmp_in_frames_dir,
            high_quality=high_quality,
            desired_fps=desired_fps
        )

        # Do a bulk swapping
        retval = _bulk(
            source_image=source_image,
            input_directory=tmp_in_frames_dir,
            output_directory=output_frames_dir,
            source_faces_index=source_faces_index,
            target_faces_index=target_faces_index,
            enhancement_options=enhancement_options,
            detection_options=detection_options,
            face_blur_options=face_blur_options,
            face_mask_correction=face_mask_correction,
            face_mask_correction_size=face_mask_correction_size,
            skip_if_exists=False,
            progressbar=progressbar
        )

        frames2video(
            video_path=output_video_file,
            input_directory=output_frames_dir,
            fps=effective_fps
        )
    finally:
        # Cleanup
        if os.path.isdir(output_frames_dir) and not keep_frames:
            shutil.rmtree(output_frames_dir)
        if os.path.isdir(tmp_in_frames_dir):
            shutil.rmtree(tmp_in_frames_dir)

    return retval
