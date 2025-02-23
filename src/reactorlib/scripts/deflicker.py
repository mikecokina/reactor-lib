import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Union, List

import cv2
import insightface
import numpy as np
from PIL import Image
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from tqdm import tqdm

from reactorlib import DetectionOptions, FaceMasker
from reactorlib.conf.settings import settings
from reactorlib.logger import suppress_output
from reactorlib.scripts.face_mask import mask_frame
from reactorlib.shared import listdir
from reactorlib.utils import create_face_helper, get_face_analyser, get_landmarks_with_face_helper


def update_ema_matrix(
        prev_matrix: np.ndarray,
        new_matrix: np.ndarray,
        alpha: float = 0.2
) -> np.ndarray:
    """
    Update the exponential moving average for the 2x3 affine matrix.
    If no previous matrix exists, return the new matrix.
    """
    if prev_matrix is None:
        return new_matrix
    return alpha * new_matrix + (1 - alpha) * prev_matrix


def landmark_based_deflicker(
        input_dir: Union[Path, str],
        output_dir: Union[Path, str],
        face_app: insightface.app.FaceAnalysis,
        face_helper: FaceRestoreHelper,
        window_size: int = 2,
        face_index: int = 0,
        use_ema: bool = False,
        ema_alpha: float = 0.2
):
    """
       Processes frames in input_dir by aligning faces using landmarks and estimating
       an affine transform. After blending the aligned frames over a sliding window,
       the blended frame is re-mapped (using the inverse of the current frame's transform)
       back to the original coordinate system of that frame.

       The output frames (as RGBA images) are saved to output_dir.
       """
    # Sanitize EMA alpha.
    ema_alpha = min(1.0, ema_alpha)

    # Get frame paths using custom listdir.
    frame_paths = listdir(input_dir)
    if not frame_paths:
        print("No frames found in input_dir")
        return None
    paths = []

    # Load frames: convert each image from PIL RGBA to OpenCV BGRA.
    frames = [
        cv2.cvtColor(np.array(Image.open(path).convert('RGBA')), cv2.COLOR_RGBA2BGRA)
        for path in tqdm(frame_paths, desc="Loading frames")
    ]

    # Use the first frame as the reference.
    ref_frame = frames[0]
    ref_landmarks = get_landmarks_with_face_helper(
        image=ref_frame,
        face_app=face_app,
        face_helper=face_helper,
        face_index=face_index,
    )
    if ref_landmarks is None:
        print("No face detected in the reference frame.")
        return None

    aligned_frames = []
    # Buffer for sliding-window temporal blending.
    frame_buffer = []
    # Variable for EMA smoothing of the affine matrix.
    prev_matrix = None

    # Process each frame.
    for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
        current_landmarks = get_landmarks_with_face_helper(
            image=frame,
            face_app=face_app,
            face_helper=face_helper,
            face_index=face_index,
        )

        # Initialize current_matrix as identity if no face is detected.
        if current_landmarks is None:
            print(f"No face detected in frame {i}; using unaligned frame.")
            current_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            aligned_frame = frame
        else:
            # Estimate affine transformation from current landmarks to reference landmarks.
            matrix, inliers = cv2.estimateAffinePartial2D(current_landmarks, ref_landmarks, method=cv2.LMEDS)
            if matrix is None:
                print(f"Transformation estimation failed for frame {i}; using unaligned frame.")
                current_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
                aligned_frame = frame
            else:
                if use_ema:
                    current_matrix = update_ema_matrix(prev_matrix, matrix, alpha=ema_alpha)
                    prev_matrix = current_matrix
                else:
                    current_matrix = matrix
                aligned_frame = cv2.warpAffine(
                    frame,
                    current_matrix,
                    (frame.shape[1], frame.shape[0]),
                    flags=cv2.INTER_LINEAR
                )

        # Add the aligned frame to the buffer.
        frame_buffer.append(aligned_frame.astype(np.float32))
        if len(frame_buffer) > window_size:
            frame_buffer.pop(0)
        # Compute the average frame over the sliding window (blended frame).
        blended_frame = np.mean(frame_buffer, axis=0).astype(np.uint8)

        # Now, move (re-map) the blended frame by applying the inverse of the current frame's transform.
        # Convert the current_matrix (2x3) to a 3x3 matrix.
        current_matrix_3x3 = np.vstack([current_matrix, [0, 0, 1]])
        inv_matrix = np.linalg.inv(current_matrix_3x3)
        inv_matrix_2x3 = inv_matrix[:2, :]  # Convert back to 2x3 for warpAffine.
        final_frame = cv2.warpAffine(
            blended_frame,
            inv_matrix_2x3,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR
        )

        aligned_frames.append(final_frame)

    # Ensure the output directory exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the final blended (and re-mapped) frames as RGBA images.
    for i, frame in enumerate(tqdm(aligned_frames, desc="Saving frames")):
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        output_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        Image.fromarray(rgba_frame).save(output_path)
        paths.append(output_path)

    return paths


def landmark_based_deflicker_(
        input_dir: Union[Path, str],
        output_dir: Union[Path, str],
        face_app: insightface.app.FaceAnalysis,
        face_helper: FaceRestoreHelper,
        window_size: int = 2,
        face_index: int = 0,
        use_ema: bool = False,
        ema_alpha: float = 0.2
) -> Union[None, List[str]]:
    """
    Processes frames in input_dir, aligns faces using all landmarks and estimates
    an affine transform. Applies temporal smoothing by averaging over a sliding
    window of frames. Optionally, smoothes the transformation matrix using
    EMA in case use_ema is True. Saves the output frames (as RGBA images) to output_dir.
    """
    # Sanitize
    ema_alpha = min(1.0, ema_alpha)

    # Get frame paths using your custom listdir.
    frame_paths = listdir(input_dir)
    paths = []

    # Load frames: convert each image from PIL RGBA to OpenCV BGRA.
    frames = [
        cv2.cvtColor(np.array(Image.open(path).convert('RGBA')), cv2.COLOR_RGBA2BGRA)
        for path in tqdm(frame_paths, desc="Loading frames")
    ]

    # Use the first frame as the reference.
    ref_frame = frames[0]
    ref_landmarks = get_landmarks_with_face_helper(
        image=ref_frame,
        face_app=face_app,
        face_helper=face_helper,
        face_index=face_index,
    )
    if ref_landmarks is None:
        return

    aligned_frames = []
    # Buffer for sliding-window temporal blending.
    frame_buffer = []
    # Variable for EMA smoothing of the affine matrix.
    prev_matrix = None

    # Process each frame.
    for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
        current_landmarks = get_landmarks_with_face_helper(
            image=frame,
            face_app=face_app,
            face_helper=face_helper,
            face_index=face_index,
        )

        # show_landmarks(Image.fromarray(frame), current_landmarks)

        if current_landmarks is None:
            print(f"No face detected in frame {i}; using unaligned frame.")
            aligned_frame = frame
        else:
            # Estimate affine transformation from current landmarks to reference landmarks.
            matrix, inliers = cv2.estimateAffinePartial2D(current_landmarks, ref_landmarks, method=cv2.LMEDS)
            if matrix is None:
                print(f"Transformation estimation failed for frame {i}; using unaligned frame.")
                aligned_frame = frame
            else:
                # Optionally smooth the transformation matrix using EMA.
                if use_ema:
                    smoothed_matrix = update_ema_matrix(prev_matrix, matrix, alpha=ema_alpha)
                    aligned_frame = cv2.warpAffine(
                        frame,
                        smoothed_matrix,
                        (frame.shape[1], frame.shape[0]),
                        flags=cv2.INTER_LINEAR
                    )
                    prev_matrix = smoothed_matrix
                else:
                    aligned_frame = cv2.warpAffine(
                        frame,
                        matrix,
                        (frame.shape[1], frame.shape[0]),
                        flags=cv2.INTER_LINEAR
                    )

        # Add the aligned frame to the buffer.
        frame_buffer.append(aligned_frame.astype(np.float32))
        # Maintain the sliding window.
        if len(frame_buffer) > window_size:
            frame_buffer.pop(0)
        # Compute the average frame over the sliding window.
        blended_frame = np.mean(frame_buffer, axis=0).astype(np.uint8)
        aligned_frames.append(blended_frame)

    # Ensure the output directory exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the final blended frames as RGBA images.
    for i, frame in enumerate(tqdm(aligned_frames, desc="Saving frames")):
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        output_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        Image.fromarray(rgba_frame).save(output_path)

        paths.append(output_path)

    return paths


def mask_frames(
        input_dir: Union[Path, str],
        output_dir: Union[Path, str],
        face_helper: FaceRestoreHelper,
        face_detection_options: DetectionOptions,
        face_index: int = 0
) -> List[str]:
    paths = []
    images = listdir(input_dir)

    for path in tqdm(images, desc="Computing masks"):
        image = Image.open(path).convert('RGB')
        image, mask = mask_frame(
            image,
            face_helper,
            face_detection_options,
            face_index=face_index
        )
        output_path = Path(output_dir) / Path(path).name
        image.save(output_path)

        paths.append(str(output_path))

    return paths


def postprocess(
        input_dir: Union[Path, str],
        deflicker_dir: Union[Path, str],
        mask_dir: Union[Path, str],
        output_dir: Union[Path, str],
):
    input_images = listdir(input_dir)
    deflicked_images = listdir(deflicker_dir)
    mask_images = listdir(mask_dir)

    for original_path, defelicked_path, mask_path in tqdm(
            zip(input_images, deflicked_images, mask_images),
            total=len(input_images),
            desc="Mergin images"
    ):
        image = Image.open(original_path).convert('RGB')
        mask = Image.open(mask_path).split()[3].convert('L')

        deflick = Image.open(defelicked_path).convert('RGB')
        result = Image.composite(deflick, image, mask)

        outputh = Path(output_dir) / Path(original_path).name
        result.save(outputh)


def deflicker(
        input_dir: Union[Path, str],
        output_dir: Union[Path, str],
        face_detection_options: DetectionOptions,
        face_index: int = 0,
):
    suffix_ = int(time.time())
    tmp_mask_dir = str(Path(tempfile.gettempdir()) / f"mask_{suffix_}")
    tmp_deflick_dir = str(Path(tempfile.gettempdir()) / f"deflick_{suffix_}")

    try:
        with suppress_output():
            face_app = get_face_analyser(det_thresh=face_detection_options.det_thresh)
            face_helper = create_face_helper()

        os.makedirs(tmp_mask_dir, exist_ok=True)
        os.makedirs(tmp_deflick_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Mask image to
        mask_frames(
            input_dir=input_dir,
            output_dir=tmp_mask_dir,
            face_helper=face_helper,
            face_index=face_index,
            face_detection_options=face_detection_options
        )

        landmark_based_deflicker(
            input_dir=tmp_mask_dir,
            output_dir=tmp_deflick_dir,
            face_app=face_app,
            face_helper=face_helper,
            face_index=face_index,
            use_ema=True,
            ema_alpha=1.0,
            window_size=2
        )

        postprocess(
            input_dir=input_dir,
            output_dir=output_dir,
            mask_dir=tmp_mask_dir,
            deflicker_dir=tmp_deflick_dir,
        )
    finally:
        if os.path.isdir(tmp_mask_dir):
            shutil.rmtree(tmp_mask_dir)
        if os.path.isdir(tmp_deflick_dir):
            shutil.rmtree(tmp_deflick_dir)


def main():
    settings.configure(**{
        'DEVICE': 'CUDA',
        "FACE_MASKER": FaceMasker.bisenet,
        "DISABLE_NSFW": False
    })

    face_detection_options = DetectionOptions(
        det_thresh=0.25,
        det_maxnum=0,
        reverse_detection_order=False
    )

    deflicker(
        input_dir="",
        output_dir="",
        face_detection_options=face_detection_options,
        face_index=0,
    )


if __name__ == '__main__':
    main()
