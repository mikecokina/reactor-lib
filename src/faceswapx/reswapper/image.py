import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from insightface.app.common import Face

from faceswapx.shared import get_warp_affine_border_value

emap = np.load(Path(os.path.dirname(__file__)) / "emap.npy")
input_std = 255.0
input_mean = 0.0


def postprocess_face(face_tensor):
    face_tensor = face_tensor.squeeze().cpu().detach()
    face_np = (face_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

    return face_np


def get_blob(aimg: np.ndarray, input_size: Tuple[int, int] = (128, 128)):
    blob = cv2.dnn.blobFromImage(
        aimg, 1.0 / input_std,
        input_size,
        (input_mean, input_mean, input_mean),
        swapRB=True
    )
    return blob


def get_latent(source_face: Face) -> np.ndarray:
    latent = source_face.normed_embedding.reshape((1, -1))
    latent = np.dot(latent, emap)
    latent /= np.linalg.norm(latent)
    return latent


def blend_swapped_image(swapped_face: np.ndarray, target_image: np.ndarray, m):
    # get image size
    h, w = target_image.shape[:2]

    # create inverse affine transform
    m_inv = cv2.invertAffineTransform(m)

    # warp swapped face back to target space
    border_value = get_warp_affine_border_value(swapped_face)
    warped_face = cv2.warpAffine(swapped_face, m_inv, (w, h), borderValue=border_value)

    # create initial white mask
    img_white = np.full(
        (swapped_face.shape[0], swapped_face.shape[1]),
        255,
        dtype=np.float32
    )

    # warp white mask to target space
    border_value = get_warp_affine_border_value(swapped_face)
    img_mask = cv2.warpAffine(img_white, m_inv, (w, h), borderValue=border_value)

    # threshold and refine mask
    img_mask[img_mask > 20] = 255

    # calculate mask size for kernel scaling
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    if len(mask_h_inds) > 0 and len(mask_w_inds) > 0:  # safety check
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h * mask_w))

        # erode mask
        k = max(mask_size // 10, 10)
        kernel = np.ones((k, k), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)

        # blur mask
        k = max(mask_size // 20, 5)
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)

    # normalize mask
    img_mask = img_mask / 255.0
    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])

    # blend images using mask
    result = img_mask * warped_face + (1 - img_mask) * target_image.astype(np.float32)
    result = result.astype(np.uint8)

    return result


def draw_keypoints(image, keypoints, color_bgr, keypoints_radius: int = 2):
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(
            image,
            (x, y),
            radius=keypoints_radius,
            color=color_bgr,
            thickness=-1
        )  # BGR format, -1 means filled circle
