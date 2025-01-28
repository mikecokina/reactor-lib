import numpy as np
from nudenet import NudeDetector

from . import settings
from .shared import SingletonBase, SharedModelKeyMixin


class NudeDetectorCache(SingletonBase, SharedModelKeyMixin):
    # todo: unify somewhere in parent
    def get_model(self) -> NudeDetector:
        instance = self._instances.get(self.__class__)
        if instance.__getattribute__('model') is None:
            instance.model = NudeDetector()
        return instance.model


NSFW_LABELS = [
    # "FEMALE_GENITALIA_COVERED",
    # "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    # "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    # "FEET_EXPOSED",
    # "BELLY_COVERED",
    # "FEET_COVERED",
    # "ARMPITS_COVERED",
    # "ARMPITS_EXPOSED",
    # "FACE_MALE",
    # "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    # "ANUS_COVERED",
    # "FEMALE_BREAST_COVERED",
    # "BUTTOCKS_COVERED",
]


nude_detector_cache = NudeDetectorCache()


class NSFWDetectedError(Exception):
    pass


def is_nsfw(image: np.ndarray, threshold: float = 0.6) -> bool:
    if not settings.DISABLE_NSFW:
        return False

    detector = nude_detector_cache.get_model()

    # NudeDetector can return detections of explicit sexual content
    result = detector.detect(image)
    # result is typically a list of dictionaries like:
    #   [{'label': 'EXPOSED_ANUS', 'score': 0.98, 'box': [x1, y1, x2, y2]}, ...]

    # We can check if any detection in the result has a score >= threshold
    for detection in result:
        # If the label is something we consider NSFW, and the confidence is above threshold,
        # we mark the image as NSFW.
        if detection['class'] in NSFW_LABELS and detection['score'] >= threshold:
            # e.g., 'EXPOSED_ANUS', 'EXPOSED_BREAST_F', 'EXPOSED_GENITALIA_F', etc.
            return True

    return False
