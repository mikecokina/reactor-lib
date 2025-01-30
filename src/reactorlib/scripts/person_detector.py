import os
from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from reactorlib.nudity import nude_detector_cache
from reactorlib.scripts.object_detection import get_object_detection_res, YOLO_PERSON_STR
from reactorlib.shared import listdir


class Gender(Enum):
    male = "male"
    female = "female"
    any = "any"


def person_detector(
        input_directory: Union[str, Path],
        output_directory: Union[str, Path],
        threshold: float = 0.6,
        gender: Gender = Gender.any
):
    def save_file(image_, file_):
        filename_ = Path(file_).name
        output_path_ = Path(output_directory, filename_)
        image_.save(output_path_)

    if gender == Gender.female:
        classes = ["FACE_FEMALE"]
    elif gender == Gender.male:
        classes = ["FACE_MALE"]
    elif gender == Gender.any:
        classes = ["FACE_MALE", "FACE_FEMALE"]
    else:
        raise ValueError('Gender must be value of Gender enum')

    detector = nude_detector_cache.get_model()

    # Create the output folder if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    files = listdir(input_directory)
    # Iterate over all images in the folder
    for file in tqdm(files):
        image = Image.open(file)
        object_detection = get_object_detection_res(image)

        if YOLO_PERSON_STR not in object_detection:
            continue

        if (YOLO_PERSON_STR in object_detection) and gender == Gender.any:
            save_file(image, file)

        detections = detector.detect(np.array(image))
        for detection in detections:
            if detection['class'] in classes and detection['score'] >= threshold:
                save_file(image, file)
                break
