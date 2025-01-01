from typing import Union, Tuple, Text

import cv2
import numpy as np
from skimage import transform as trans

from reactorlib.shared import get_warp_affine_border_value

arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ],
    dtype=np.float32
)


# noinspection PyUnusedLocal
def estimate_norm(lmk: np.ndarray, image_size: int = 112, mode='arcface') -> np.ndarray:
    # assert lmk.shape == (5, 2)
    # assert image_size%112==0 or image_size%128==0
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x

    # offset = 0.0039 * resolution - 0.5

    if image_size == 160:
        dst[:, 0] += 0.1
        dst[:, 1] += 0.1
    elif image_size == 256:
        dst[:, 0] += 0.5
        dst[:, 1] += 0.5
    elif image_size == 320:
        dst[:, 0] += 0.75
        dst[:, 1] += 0.75
    elif image_size == 512:
        dst[:, 0] += 1.5
        dst[:, 1] += 1.5

    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    m = tform.params[0:2, :]
    return m


def norm_crop(
        img: np.ndarray,
        landmark,
        image_size: int = 112,
        mode='arcface'
):
    m = estimate_norm(landmark, image_size, mode)
    border_value = get_warp_affine_border_value(img)
    warped = cv2.warpAffine(img, m, (image_size, image_size), borderValue=border_value)
    return warped


def norm_crop2(img: np.ndarray, landmark, image_size: int = 112, mode: Text = 'arcface'):
    m = estimate_norm(landmark, image_size, mode)
    border_value = get_warp_affine_border_value(img)
    warped = cv2.warpAffine(img, m, (image_size, image_size), borderValue=border_value)
    return warped, m


def square_crop(im: np.ndarray, s):
    if im.shape[0] > im.shape[1]:
        height = s
        width = int(float(im.shape[1]) / im.shape[0] * s)
        scale = float(s) / im.shape[0]
    else:
        width = s
        height = int(float(im.shape[0]) / im.shape[1] * s)
        scale = float(s) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((s, s, 3), dtype=np.uint8)
    det_im[:resized_im.shape[0], :resized_im.shape[1], :] = resized_im
    return det_im, scale


def transform(data: np.ndarray, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    # translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    m = t.params[0:2]

    border_value = get_warp_affine_border_value(data)
    cropped = cv2.warpAffine(data, m, (output_size, output_size), borderValue=border_value)
    return cropped, m


def trans_points2d(pts, m):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(m, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, m):
    scale = np.sqrt(m[0][0] * m[0][0] + m[0][1] * m[0][1])
    # print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(m, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, m):
    if pts.shape[1] == 2:
        return trans_points2d(pts, m)
    else:
        return trans_points3d(pts, m)
