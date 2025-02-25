import os
from typing import Tuple, Union

import cv2
import insightface
import numpy as np
from PIL import Image
from facexlib.detection import retinaface
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from .conf.settings import settings


def create_face_helper() -> FaceRestoreHelper:
    if hasattr(retinaface, 'device'):
        retinaface.device = settings.device

    return FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=settings.device,
        model_rootpath=os.path.join(settings.MODELS_PATH, 'facexlib')
    )


def get_face_analyser(
    det_thresh: float = 0.5,
    det_size: Tuple[int, int] = (640, 640)
) -> insightface.app.FaceAnalysis:
    # Initialize InsightFace's face analysis app.
    face_app = insightface.app.FaceAnalysis(
        root=os.path.join(settings.MODELS_PATH, 'insightface'),
    )
    face_app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)
    return face_app


def show_landmarks(image: Image.Image, landmarks: np.ndarray) -> None:
    np_image = np.array(image.convert('RGB'), dtype=np.uint8)[:, :, ::-1]
    image_bgr_vis = np_image.copy()
    for (x, y) in landmarks.astype(np.int32):
        cv2.circle(image_bgr_vis, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

    cv2.imshow("Landmarks on Original Image", image_bgr_vis)

    # cv2.imwrite("/home/mike/Data/reels/174/landmarks/" + f"frame_{i:04d}.png", image_bgr_vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_landmarks_2d_106(
        image: np.ndarray,
        face_app: insightface.app.FaceAnalysis
) -> Union[None, np.ndarray]:
    """
    Convert the BGRA image to RGB and use InsightFace to detect a face.
    If detected, return all available landmarks.
    Preference is given to landmark_2d_106; if not available, fallback to landmark_3d_68.
    """
    # Convert from BGRA to RGB for InsightFace
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    faces = face_app.get(image_rgb)
    if len(faces) == 0:
        return None
    face = faces[0]
    if face.landmark_2d_106 is not None:
        return face.landmark_2d_106.astype(np.float32)
    else:
        return None


def get_landmarks_with_face_helper(
        image: Union[np.ndarray, Image.Image],
        face_helper: FaceRestoreHelper,
        face_app: insightface.app.FaceAnalysis,
        face_index: int = 0
) -> Union[None, np.ndarray]:
    if isinstance(image, Image.Image):
        if image.mode == 'RGBA':
            image_alpha = image.split()[3]
            image_rgb = image.convert('RGB')
            background = Image.new('RGBA', image_rgb.size, (255, 255, 255, 0))
            image = Image.composite(image_rgb, background, image_alpha)

        image = image.convert('RGB')
        np_image = np.array(image, dtype=np.uint8)[:, :, ::-1]
    else:
        np_image = image

    face_helper.clean_all()
    face_helper.read_image(np_image)
    face_helper.get_face_landmarks_5(
        only_center_face=False,
        resize=640,
        eye_dist_threshold=5
    )
    face_helper.align_warp_face()

    # Make sure index for face detected
    if not (
            face_helper.cropped_faces is not None
            and len(face_helper.cropped_faces) <= face_index + 1
    ):
        return None

    cropped_face = face_helper.cropped_faces[face_index]
    landmarks_cropped = get_landmarks_2d_106(cropped_face, face_app)

    # Return None if no landmarks detected
    if landmarks_cropped is None:
        return landmarks_cropped

    # Retrieve the inverse affine transformation matrix from face_helper.
    face_helper.get_inverse_affine(None)
    inv_affine = face_helper.inverse_affine_matrices[0]

    # Convert landmark coordinates to homogeneous coordinates for the transformation.
    ones = np.ones((landmarks_cropped.shape[0], 1), dtype=np.float32)
    landmarks_hom = np.concatenate([landmarks_cropped, ones], axis=1)  # shape: (N, 3)

    # Apply the inverse affine transformation to map points back to original image coordinates.
    landmarks_original = (inv_affine @ landmarks_hom.T).T  # shape: (N, 2)

    return landmarks_original
