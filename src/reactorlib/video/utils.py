import os

from pathlib import Path
from typing import Union, Tuple

import cv2

from ..shared import listdir, GradioTqdmWrapper, get_tqdm_cls


def video2frames(
        video_path: Union[Path, str],
        output_directory: Union[Path, str],
        high_quality: bool = True,
        progressbar: bool = True,
        desired_fps: float = None,
        **kwargs

) -> Tuple[int, float]:
    tqdm = get_tqdm_cls(progressbar=progressbar)
    if kwargs.get("gr_progressbar"):
        tqdm = GradioTqdmWrapper(kwargs.get("gr_progressbar"))

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise OSError(f"Error: Could not open video file '{video_path}'. Check if the file exists and is accessible.")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Decide which FPS we'll actually use
    if (desired_fps is None) or (desired_fps >= original_fps):
        # Use original fps if desired_fps is not provided or is >= original_fps
        output_fps = original_fps
    else:
        # Use desired_fps, but we need to skip frames to achieve this
        output_fps = desired_fps

    # If output_fps == original_fps, skip_factor = 1 => no skipping
    # Otherwise, skip_factor is roughly how many frames we skip for each saved frame.
    # This is a simple approach; it won't match the desired FPS exactly if there's
    # not an integer ratio, but it's straightforward and fast.
    if output_fps == 0:
        raise ValueError("The video FPS is 0, or desired_fps is 0. Unable to process.")

    skip_factor = int(round(original_fps / output_fps))
    if skip_factor < 1:
        skip_factor = 1  # Just a safeguard.

    os.makedirs(output_directory, exist_ok=True)

    frame_index = 0
    saved_frames = 0

    with tqdm(total=total_frames, desc="Extracting Frames", unit="frames") as pbar:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Only save the frame if we are on the correct multiple
            # (meaning we effectively "skip" others).
            if frame_index % skip_factor == 0:
                saved_frames += 1
                output_path = Path(output_directory) / f"frame_{saved_frames:06d}"

                if high_quality:
                    output_file = str(output_path) + ".png"
                    cv2.imwrite(output_file, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                else:
                    output_file = str(output_path) + ".jpg"
                    cv2.imwrite(output_file, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            frame_index += 1
            pbar.update(1)

    cap.release()

    # Compute the "effective" FPS based on how many frames we actually saved
    # and the total duration of the original video.
    duration_seconds = total_frames / original_fps if original_fps > 0 else 1e-6
    effective_fps = round(saved_frames / duration_seconds * 100) / 100 if duration_seconds > 0 else 0

    return saved_frames, effective_fps


def frames2video(
        video_path: Union[Path, str],
        input_directory: Union[Path, str],
        fps: Union[float, int],
        progressbar: bool = True,
        **kwargs
):
    tqdm = get_tqdm_cls(progressbar=progressbar)
    if kwargs.get("gr_progressbar"):
        tqdm = GradioTqdmWrapper(kwargs.get("gr_progressbar"))

    frame_files = listdir(input_directory)

    if not frame_files:
        raise FileNotFoundError("No frames found in the directory!")

    # Read the first frame to get the dimensions
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape

    # Choose a FOURCC code. 'mp4v' often works for MP4 container, but
    # advanced control over H.264 is limited.
    # noinspection PyUnresolvedReferences
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(video_path),  # Output path
        fourcc,
        fps,
        (width, height),
    )

    if not video_writer.isOpened():
        raise OSError("Could not open VideoWriter with the chosen codec.")

    # Write frames
    with tqdm(total=len(frame_files), desc="Rendering Video", unit="frames") as pbar:
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            if frame is not None:
                video_writer.write(frame)
            pbar.update(1)
    video_writer.release()
