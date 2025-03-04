import cv2
import numpy as np
import torch


class MaskGeneratorMixin(object):
    @staticmethod
    def morph_mask(mask: np.ndarray, mask_size: int):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        if mask_size > 0:
            mask = cv2.dilate(mask, kernel, iterations=mask_size)
        elif mask_size < 0:
            mask = cv2.erode(mask, kernel, iterations=abs(mask_size))
        return mask


class TorchMaskGeneratorMixin(object):
    def get_model(self) -> torch.nn.Module:
        if hasattr(self, 'model'):
            return self.model
        raise NotImplementedError("Mixin requires to be part of class that implemnets torch `model`")

    def get_device(self):
        # todo: imlpement device id option
        _device = next(self.get_model().parameters()).device
        device = _device.type
        # index = _device.index
        return device

    def set_device(self, device: str):
        self.get_model().to(device)
