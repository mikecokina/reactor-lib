import os
import shutil
import tempfile
import time
from pathlib import Path

import gradio as gr
from PIL import Image

from faceswapx import (
    swap,
    EnhancementOptions,
    FaceEnhancementOptions,
    DetectionOptions,
    FaceSwapper,
    FaceMasker,
    settings, FaceBlurOptions
)
from faceswapx.react import _bulk
from faceswapx.shared import torch_gc
from faceswapx.video.utils import video2frames, frames2video

IS_RUNNING = False


class Terminator(Exception):
    pass


def toogle_is_running():
    global IS_RUNNING
    IS_RUNNING = not IS_RUNNING


# noinspection DuplicatedCode
def operate(
        source_image: Image.Image,
        target_video: str,
        output_directory: str,
        progress=gr.Progress(track_tqdm=True),
        **kwargs
) -> str:
    global IS_RUNNING
    IS_RUNNING = True

    enhancement_options = EnhancementOptions(
        face_enhancement_options=FaceEnhancementOptions(
            do_enhancement=kwargs.get("do_enhancement", True),
            enhance_target=kwargs.get("enhance_target", False),
            codeformer_visibility=kwargs.get("codeformer_visibility", 0.5),
            codeformer_weight=kwargs.get("codeformer_weight", 0.5),
            restore_face_only=kwargs.get("restore_face_only", False),
            face_detection_options=DetectionOptions(
                det_thresh=0.25,
                det_maxnum=0
            )
        ),
    )

    detection_options = DetectionOptions(
        det_thresh=kwargs.get("det_thresh", 0.65),
        det_maxnum=int(kwargs.get("det_maxnum", 0))
    )
    keep_frames = bool(kwargs.get("keep_frames", False))

    face_swapper = FaceSwapper(kwargs.get('face_swapper', FaceSwapper.inswapper_128.value))
    face_masker = FaceMasker(kwargs.get('face_masker', FaceMasker.bisenet.value))

    face_source_index = int(kwargs.get("face_source_index", 0))
    face_target_index = int(kwargs.get("face_target_index", 0))

    device = kwargs.get("device_option", "CUDA")
    settings.configure(**{
        'DEVICE': device,
        "FACE_SWAPPER": face_swapper,
        "FACE_MASKER": face_masker,
        "DISABLE_NSFW": os.environ.get("FACESWAPX_DISABLE_NSFW", True)
    })

    # Logic
    suffix_ = int(time.time())
    tmp_in_frames_dir = str(Path(tempfile.gettempdir()) / f"frames_in_{suffix_}")
    output_frames_dir = str(Path(output_directory) / f"frames_out_{suffix_}")
    output_video_file = Path(output_directory) / f"{Path(target_video).stem}_swapped.mp4"
    os.makedirs(output_directory, exist_ok=True)

    try:
        _, effective_fps = video2frames(
            video_path=target_video,
            output_directory=tmp_in_frames_dir,
            high_quality=True,
            desired_fps=25,
            gr_progressbar=progress.tqdm
        )

        # Do a bulk swapping
        _bulk(
            source_image=source_image,
            input_directory=tmp_in_frames_dir,
            output_directory=output_frames_dir,
            source_faces_index=[face_source_index],
            target_faces_index=[face_target_index],
            enhancement_options=enhancement_options,
            detection_options=detection_options,
            face_mask_correction=True,
            face_mask_correction_size=10,
            skip_if_exists=False,
            progressbar=True,
            face_blur_options=FaceBlurOptions(),
            gr_progressbar=progress.tqdm,
            start_frame=1,
            end_frame=None
        )

        frames2video(
            video_path=output_video_file,
            input_directory=output_frames_dir,
            fps=effective_fps,
            gr_progressbar=progress.tqdm
        )

    except Terminator:
        pass
    finally:
        # Cleanup
        if os.path.isdir(output_frames_dir) and not keep_frames:
            shutil.rmtree(output_frames_dir)
        if os.path.isdir(tmp_in_frames_dir):
            shutil.rmtree(tmp_in_frames_dir)

        toogle_is_running()
        torch_gc()

    return "Completed successfully!"


def update_button(source_image, target_video, output_directory):
    # Check if both images are provided
    if ((source_image is not None)
            and (target_video is not None)
            and isinstance(output_directory, str)
            and len(output_directory) > 3):
        # Enable button and change color
        return gr.update(interactive=True, variant="primary")
    else:
        # Disable button and reset to default color
        return gr.update(interactive=False, variant="secondary")


# noinspection DuplicatedCode
def main():
    with gr.Blocks() as demo:
        gr.Markdown("### Face Enhancement Options")
        with gr.Row():
            do_enhancement = gr.Checkbox(label="do_enhancement", value=True)
            restore_face_only = gr.Checkbox(label="restore_face_only", value=True)
            codeformer_visibility = gr.Slider(value=0.5, minimum=0, maximum=1, step=0.05, label="Codeformer Visibility")
            codeformer_weight = gr.Slider(value=0.5, minimum=0, maximum=1, step=0.05, label="Codeformer Weight")

        gr.Markdown("### Detection Options")
        with gr.Row():
            det_thresh = gr.Slider(value=0.65, minimum=0, maximum=1, step=0.05, label="Face Detection Thresh")
            det_maxnum = gr.Number(value=0, minimum=0, maximum=5, label="Maximum Faces Detected")
            face_source_index = gr.Number(value=0, minimum=0, maximum=5, label="Source Face Index")
            face_target_index = gr.Number(value=0, minimum=0, maximum=5, label="Target Face Index")

        gr.Markdown("### Models Selection")
        with gr.Row():
            face_swapper = gr.Dropdown(
                label="Face Swapper",
                choices=[
                    FaceSwapper.inswapper_128.value,
                    FaceSwapper.inswapper_256.value,
                    FaceSwapper.inswapper_512.value,
                    FaceSwapper.reswapper_128.value,
                    FaceSwapper.reswapper_256_1567500.value
                ])

            face_masker = gr.Dropdown(
                label="Face Masker",
                choices=[
                    FaceMasker.bisenet.value,
                    FaceMasker.birefnet_large.value,
                    FaceMasker.birefnet_tiny.value
                ])

            device_option = gr.Dropdown(
                label="Device",
                choices=["CUDA", "CPU"]
            )

        gr.Markdown("### Vieo Processor Options")
        with gr.Row():
            keep_frames = gr.Checkbox(label="keep_frames", value=False)
            output_directory = gr.Text(label="Output Directory")

        # gr.Markdown("### Input Images")
        with gr.Row():  # Create a row for the inputs
            source_image = gr.Image(type="pil", label="Source Image", width=300, height=400)
            target_video = gr.Video(label="Target Video", width=300, height=400)
            with gr.Column():
                progress = gr.Text(label="Progress", interactive=False)

        inputs = {
            "det_thresh": det_thresh,
            "det_maxnum": det_maxnum,
            "restore_face_only": restore_face_only,
            "do_enhancement": do_enhancement,
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
            "source_image": source_image,
            "target_video": target_video,
            "keep_frames": keep_frames,
            "output_directory": output_directory,
            "face_swapper": face_swapper,
            "face_masker": face_masker,
            "face_source_index": face_source_index,
            "face_target_index": face_target_index,
            "device_option": device_option,
        }

        with gr.Row():
            # Connect the function
            btn_run = gr.Button("Run", interactive=False, variant="secondary")
            btn_run.click(
                fn=lambda *args: operate(
                    **{key: value for key, value in zip(inputs.keys(), args)}  # Convert positional args to kwargs
                ),
                inputs=list(inputs.values()),
                outputs=progress
            )

            btn_terminate = gr.Button("Terminate", interactive=True, variant="stop")
            # noinspection PyTypeChecker
            btn_terminate.click(
                fn=toogle_is_running,
                inputs=None,
                outputs=None,
            )

        # Dynamically update the button state and style when images change
        instances = [source_image, target_video, output_directory]

        for instance in instances:
            instance.change(
                update_button,
                inputs=instances,
                outputs=btn_run
            )

    demo.launch(share=False)


if __name__ == '__main__':
    main()
