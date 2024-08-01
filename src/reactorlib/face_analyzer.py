import copy
import os
from typing import Union, Tuple

import numpy as np
from PIL import Image

from . import settings
from .logger import suppress_output
from .modloader import get_analysis_model
from .logger import logger


def analyze(
        image: Image.Image | str,
        det_size=(640, 640),
        det_thresh=0.5,
        det_maxnum=0
):
    import cv2
    from . import images

    img = images.get_image(image)
    # noinspection PyTypeChecker
    source_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return analyze_faces(
        source_img,
        det_size=det_size,
        det_thresh=det_thresh,
        det_maxnum=det_maxnum
    )


def analyze_faces(
        img_data: np.ndarray,
        det_size=(640, 640),
        det_thresh=0.5,
        det_maxnum=0
):
    with suppress_output():
        face_analyser = copy.deepcopy(get_analysis_model(settings.MODELS_PATH))
        face_analyser.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)
        return face_analyser.get(img_data, max_num=det_maxnum)


def _get_face_gender_logic(face, face_index, gender_condition, operated, face_gender):
    logger.info("%s Face %s: Detected Gender -%s-", operated, face_index, face_gender)
    if (gender_condition == 1 and face_gender == "F") or (gender_condition == 2 and face_gender == "M"):
        logger.info("OK - Detected Gender matches Condition")
        try:
            return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
        except IndexError:
            return None, 0
    else:
        logger.info("WRONG - Detected Gender doesn't match Condition")
        return sorted(face, key=lambda x: x.bbox[0])[face_index], 1


def get_face_gender(
        face,
        face_index,
        gender_condition,
        operated: str,
        gender_detected,
        _suppress_output: bool = False
):
    face_gender = gender_detected
    if face_gender == "None":
        return None, 0

    if _suppress_output:
        with suppress_output():
            return _get_face_gender_logic(face, face_index, gender_condition, operated, face_gender)
    else:
        return _get_face_gender_logic(face, face_index, gender_condition, operated, face_gender)


def get_face_age(face, face_index) -> Union[int, str]:
    age = [x.age for x in face]
    age.reverse()
    # noinspection PyBroadException
    try:
        face_age = age[face_index]
    except Exception:
        logger.warning(f"Age detection: No face with index = {str(face_index)} was found")
        return "None"
    return face_age


def get_gender(face, face_index) -> Union[str]:
    gender = [x.sex for x in face]
    gender.reverse()
    # noinspection PyBroadException
    try:
        face_gender = gender[face_index]
    except Exception:
        logger.warning(f"Gender detection: No face with index = {str(face_index)} was found")
        return "None"
    return face_gender


def _half_det_size(det_size) -> Tuple[float, float]:
    logger.info("Trying to halve 'det_size' parameter")
    return det_size[0] // 2, det_size[1] // 2


def get_face_single(
        *,
        img_data: np.ndarray,
        face,
        face_index=0,
        det_size=(640, 640),
        gender_source=0,
        gender_target=0,
        det_thresh=0.5,
        det_maxnum=0
) -> Tuple:
    buffalo_path = os.path.join(settings.MODELS_PATH, "insightface/models/buffalo_l.zip")
    if os.path.exists(buffalo_path):
        os.remove(buffalo_path)

    face_age = "None"
    face_gender = "None"
    gender_detected = "None"

    # noinspection PyBroadException
    try:
        face_age = get_face_age(face, face_index)
    except Exception:
        logger.warning(f"Cannot detect any age for face index = {face_index}")

    # noinspection PyBroadException
    try:
        face_gender = get_gender(face, face_index)
        gender_detected = face_gender
        face_gender = "Female" if face_gender == "F" else ("Male" if face_gender == "M" else "None")
    except Exception:
        logger.warning(f"Cannot detect any Gender for Face index = %s", face_index)

    if gender_source != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = _half_det_size(det_size)
            face_ = analyze_faces(img_data, det_size_half, det_thresh, det_maxnum)
            return get_face_single(
                img_data=img_data,
                face=face_,
                face_index=face_index,
                det_size=det_size_half,
                gender_source=gender_source,
                gender_target=gender_target,
                det_thresh=det_thresh,
                det_maxnum=det_maxnum
            )
        faces, wrong_gender = get_face_gender(face, face_index, gender_source, "Source", gender_detected)
        return faces, wrong_gender, face_age, face_gender

    if gender_target != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = _half_det_size(det_size)
            face_ = analyze_faces(img_data, det_size_half, det_thresh, det_maxnum)
            return get_face_single(
                img_data=img_data,
                face=face_,
                face_index=face_index,
                det_size=det_size_half,
                gender_source=gender_source,
                gender_target=gender_target,
                det_thresh=det_thresh,
                det_maxnum=det_maxnum
            )

        faces, wrong_gender = get_face_gender(face, face_index, gender_target, "Target", gender_detected)
        return faces, wrong_gender, face_age, face_gender

    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = _half_det_size(det_size)
        face_ = analyze_faces(img_data, det_size_half, det_thresh, det_maxnum)
        return get_face_single(
            img_data=img_data,
            face=face_,
            face_index=face_index,
            det_size=det_size_half,
            gender_source=gender_source,
            gender_target=gender_target,
            det_thresh=det_thresh,
            det_maxnum=det_maxnum
        )

    try:
        return sorted(face, key=lambda x: x.bbox[0])[face_index], 0, face_age, face_gender
    except IndexError:
        return None, 0, face_age, face_gender
