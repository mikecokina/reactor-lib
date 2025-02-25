import cv2
import numpy as np


class MaskGeneratorMixin(object):
    @staticmethod
    def morph_mask(mask: np.ndarray, mask_size: int):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        if mask_size > 0:
            mask = cv2.dilate(mask, kernel, iterations=mask_size)
        elif mask_size < 0:
            mask = cv2.erode(mask, kernel, iterations=abs(mask_size))
        return mask
