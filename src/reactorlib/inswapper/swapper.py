import os
from typing import Text, Tuple, Union

import cv2
import numpy as np
import onnx
import onnxruntime

from onnx import numpy_helper
from insightface.app.common import Face
from insightface.utils import face_align


from ..conf.settings import settings
from ..conf.settings import FaceSwapper


class INSwapper(object):
    def __init__(self, model_path: Text, session=None):
        model_path = model_path or os.path.join(settings.MODELS_PATH, settings.FACE_SWAPPER_MODEL_DOWNLOAD_NAME)

        self.model_file = model_path
        self.session = session
        model = onnx.load(str(self.model_file))
        graph = model.graph
        self.emap = numpy_helper.to_array(graph.initializer[-1])
        self.input_mean = 0.0
        self.input_std = 255.0

        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)

        inputs = self.session.get_inputs()
        self.input_names = []

        for inp in inputs:
            self.input_names.append(inp.name)

        outputs = self.session.get_outputs()
        output_names = []

        for out in outputs:
            output_names.append(out.name)

        self.output_names = output_names
        assert len(self.output_names) == 1

        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        self.input_shape = input_shape
        self.input_size = tuple(input_shape[2:4][::-1])

    @staticmethod
    def _paste_back(
            image: np.ndarray,
            cropped: np.ndarray,
            bgr_fake: np.ndarray,
            norm: np.ndarray
    ) -> np.ndarray:
        target_img = image
        fake_diff = bgr_fake.astype(np.float32) - cropped.astype(np.float32)
        fake_diff = np.abs(fake_diff).mean(axis=2)
        fake_diff[:2, :] = 0
        fake_diff[-2:, :] = 0
        fake_diff[:, :2] = 0
        fake_diff[:, -2:] = 0
        im = cv2.invertAffineTransform(norm)
        img_white = np.full((cropped.shape[0], cropped.shape[1]), 255, dtype=np.float32)
        # noinspection PyTypeChecker
        bgr_fake = cv2.warpAffine(bgr_fake, im, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
        # noinspection PyTypeChecker
        img_white = cv2.warpAffine(img_white, im, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
        # noinspection PyTypeChecker
        fake_diff = cv2.warpAffine(fake_diff, im, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
        img_white[img_white > 20] = 255
        fthresh = 10
        fake_diff[fake_diff < fthresh] = 0
        fake_diff[fake_diff >= fthresh] = 255
        img_mask = img_white
        mask_h_inds, mask_w_inds = np.where(img_mask == 255)
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h * mask_w))
        k = max(mask_size // 10, 10)

        kernel = np.ones((k, k), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        kernel = np.ones((2, 2), np.uint8)
        fake_diff = cv2.dilate(fake_diff, kernel, iterations=1)
        k = max(mask_size // 20, 5)

        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
        k = 5
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
        img_mask /= 255
        fake_diff /= 255

        img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
        fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(np.float32)
        fake_merged = fake_merged.astype(np.uint8)
        return fake_merged

    def _swap_face(
            self,
            image: np.ndarray,
            target_face: Face,
            source_face: Face,
            dim: int = 2,
            paste_back: bool = True
    ):
        size = int(128 * dim)
        norm = face_align.estimate_norm(target_face.kps, size, 'arcface')
        aimg, _ = face_align.norm_crop2(image, target_face.kps, size)

        output = np.zeros((size, size, 3), dtype=np.float32)

        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)

        # Fake hi-res inswapper
        for j in range(dim):
            for i in range(dim):
                aimg_shifted = aimg[j::dim, i::dim]

                blob = cv2.dnn.blobFromImage(
                    aimg_shifted,
                    1.0 / self.input_std,
                    self.input_size,
                    (self.input_mean, self.input_mean, self.input_mean),
                    swapRB=True
                )
                pred = self.session.run(
                    self.output_names,
                    {
                        self.input_names[0]: blob,
                        self.input_names[1]: latent
                    })[0]
                img_fake = pred.transpose((0, 2, 3, 1))[0]
                bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]

                output[j::dim, i::dim] = bgr_fake.copy()

        if not paste_back:
            return output, norm
        else:
            return self._paste_back(image, aimg, output, norm)

    def __swap_face(
            self,
            image: np.ndarray,
            target_face: Face,
            source_face: Face,
            paste_back: bool = True,
    ) -> Union[Tuple[np.ndarray, Face], np.ndarray]:
        aimg, norm = face_align.norm_crop2(image, target_face.kps, self.input_size[0])
        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / self.input_std,
            self.input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True
        )
        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        pred = self.session.run(self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent})[0]

        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
        if not paste_back:
            return bgr_fake, norm
        else:
            return self._paste_back(image, aimg, bgr_fake, norm)

    def get(
            self,
            result: np.ndarray,
            target_face: Face,
            source_face: Face,
            paste_back=True,
    ) -> np.ndarray:
        dim = 1
        if settings.FACE_SWAPPER == FaceSwapper.inswapper_256:
            dim = 2
        elif settings.FACE_SWAPPER == FaceSwapper.inswapper_512:
            dim = 4

        return self._swap_face(
            result,
            target_face,
            source_face,
            dim,
            paste_back,
        )
