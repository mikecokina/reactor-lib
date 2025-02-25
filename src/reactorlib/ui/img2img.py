import os

import gradio as gr
from PIL import Image

from reactorlib import (
    swap,
    EnhancementOptions,
    FaceEnhancementOptions,
    DetectionOptions,
    FaceSwapper,
    FaceMasker,
    settings
)
from reactorlib.shared import torch_gc


def operate(
        source_image: Image.Image,
        target_image: Image.Image,
        **kwargs
) -> Image.Image:
    enhancement_options = EnhancementOptions(
        face_enhancement_options=FaceEnhancementOptions(
            do_enhancement=kwargs.get("do_enhancement", True),
            enhance_target=kwargs.get("enhance_target", False),
            codeformer_visibility=kwargs.get("codeformer_visibility", 0.5),
            codeformer_weight=kwargs.get("codeformer_weight", 0.5),
            restore_face_only=kwargs.get("restore_face_only", False),
            face_detection_options=DetectionOptions(
                det_thresh=0.25,
                det_maxnum=0,
                mask_size=-1,
                mask_blur_kernel=33,
                mask_from_source=True
            )
        ),
    )
    detection_options = DetectionOptions(
        det_thresh=kwargs.get("det_thresh", 0.65),
        det_maxnum=int(kwargs.get("det_maxnum", 0))
    )

    face_swapper = FaceSwapper(kwargs.get('face_swapper', FaceSwapper.inswapper_128.value))
    face_masker = FaceMasker(kwargs.get('face_masker', FaceMasker.bisenet.value))

    face_source_index = int(kwargs.get("face_source_index", 0))
    face_target_index = int(kwargs.get("face_target_index", 0))

    device = kwargs.get("device_option", "CUDA")
    settings.configure(**{
        'DEVICE': device,
        "FACE_SWAPPER": face_swapper,
        "FACE_MASKER": face_masker,
        "DISABLE_NSFW": os.environ.get("REACTORLIB_DISABLE_NSFW", True)
    })

    result_image, _ = swap(
        source_image=source_image,
        target_image=target_image,
        source_faces_index=[face_source_index],
        target_faces_index=[face_target_index],
        enhancement_options=enhancement_options,
        detection_options=detection_options,
        face_mask_correction=True,
        face_mask_correction_size=10
    )

    torch_gc()
    return result_image


def update_button(source_image, target_image):
    # Check if both images are provided
    if source_image is not None and target_image is not None:
        # Enable button and change color
        return gr.update(interactive=True, variant="primary")
    else:
        # Disable button and reset to default color
        return gr.update(interactive=False, variant="secondary")


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### Face Enhancement Options")
        with gr.Row():
            restore_face_only = gr.Checkbox(label="restore_face_only", value=True)
            do_enhancement = gr.Checkbox(label="do_enhancement", value=True)
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

        gr.Markdown("### Input Images")
        with gr.Row():  # Create a row for the inputs
            source_image = gr.Image(type="pil", label="Source Image", width=300, height=400)
            target_image = gr.Image(type="pil", label="Target Image", width=300, height=400)
            result = gr.Image(label="Result", width=300, height=400)

        inputs = {
            "det_thresh": det_thresh,
            "det_maxnum": det_maxnum,
            "restore_face_only": restore_face_only,
            "do_enhancement": do_enhancement,
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
            "source_image": source_image,
            "target_image": target_image,
            "face_swapper": face_swapper,
            "face_masker": face_masker,
            "face_source_index": face_source_index,
            "face_target_index": face_target_index,
            "device_option": device_option
        }

        # Connect the function
        btn = gr.Button("Run", interactive=False, variant="secondary")
        btn.click(
            fn=lambda *args: operate(
                **{key: value for key, value in zip(inputs.keys(), args)}  # Convert positional args to kwargs
            ),
            inputs=list(inputs.values()),
            outputs=result
        )

        # Dynamically update the button state and style when images change
        source_image.change(
            update_button,
            inputs=[source_image, target_image],
            outputs=btn
        )

        target_image.change(
            update_button,
            inputs=[source_image, target_image],
            outputs=btn
        )

    demo.launch(share=False)


if __name__ == '__main__':
    main()
