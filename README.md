[![GitHub version](https://img.shields.io/badge/version-0.1.dev0-yellow.svg)](https://github.com/mikecokina/reactor-lib)
[![Licence GPLv3](https://img.shields.io/badge/license-GNU/GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
[![Python version](https://img.shields.io/badge/python-3.10|3.11-orange.svg)](https://www.python.org/doc/versions/)
[![OS](https://img.shields.io/badge/os-Linux-magenta.svg)](https://www.gnu.org/gnu/linux-and-gnu.html)


# reactor-lib

The face swap python library based on sd-webui-reactor extension

# Licence

- All repo code is published under GPLv3 Licence
- When using `FaceSwapper.inswapper` setting, please follow these  [licence conditions](https://github.com/deepinsight/insightface?tab=readme-ov-file#license).
- If you are using any ReSwapper models, please follow these [licence conditions](https://github.com/somanchiu/ReSwapper)

## Acknowledgement
- *sd-webui-reactor*: https://github.com/Gourieff/sd-webui-reactor
- *inswapper*: https://github.com/haofanwang/inswapper
- *ReSwapper*: https://github.com/somanchiu/ReSwapper
- *BiRefNet*: https://github.com/ZhengPeng7/BiRefNet


## Install

`pip install git+https://github.com/mikecokina/reactor-lib.git@master`


### onnxruntime for CUDA 12.X
pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-gpu

## Example for single image

### User Interface

"A simple image-to-image swap interface is provided in the file `ui/img2img.py`. Run it as a Python script. First, install gradio==5.13.1."

### Code

```python
from reactorlib import (
    settings, swap, DetectionOptions, EnhancementOptions, 
    FaceBlurOptions, FaceEnhancementOptions, FaceSwapper, FaceMasker
)


def main():
    settings.configure(**{
        'DEVICE': 'CUDA',
        # "MODELS_PATH": "/absolute/path/to/models/directory",
        "FACE_SWAPPER": FaceSwapper.reswapper_256_1567500,
        "FACE_MASKER": FaceMasker.birefnet
    })

    enhancement_options = EnhancementOptions(
        face_enhancement_options=FaceEnhancementOptions(
            do_enhancement=True,
            enhance_target=False,
            codeformer_visibility=1.0,
            codeformer_weight=0.5,
            restore_face_only=False,
            # Face enhancer detection options
            detection_options=DetectionOptions(
                det_thresh=0.25,
                det_maxnum=0
            )
        )
    )
    # Face swapper detection options
    detection_options = DetectionOptions(det_thresh=0.65, det_maxnum=0)
    face_blur_options = FaceBlurOptions(
        do_face_blur=False,
        do_video_noise=False,
        blur_radius=2,
        blur_strength=0.2,
        noise_pixel_size=1
    )

    result_image, n_swapped = swap(
        source_image="/absolute/path/to/source/image.<ext>",
        target_image="/absolute/path/to/target/image.<ext>",
        target_faces_index=[0],
        source_faces_index=[0],
        enhancement_options=enhancement_options,
        detection_options=detection_options,
        face_blur_options=face_blur_options
    )


if __name__ == "__main__":
    main()
```


## Example for batch processing from single source image

```python
from reactorlib import settings, swap, DetectionOptions, EnhancementOptions, FaceBlurOptions, FaceEnhancementOptions


def main():
    settings.configure(**{
        'DEVICE': 'CUDA',
    })

    enhancement_options = EnhancementOptions(
        face_enhancement_options=FaceEnhancementOptions(
            do_enhancement=True,
            enhance_target=False,
            codeformer_visibility=1.0,
            codeformer_weight=0.5,
            restore_face_only=False,
            # Face enhancer detection options for face masking 
            detection_options=DetectionOptions(
                det_thresh=0.25,
                det_maxnum=0
            )
        )
    )
    detection_options = DetectionOptions(det_thresh=0.65, det_maxnum=0)
    face_blur_options = FaceBlurOptions(
        do_face_blur=False,
        do_video_noise=False,
        blur_radius=2,
        blur_strength=0.2,
        noise_pixel_size=1
    )


    _, n_swapped = swap(
        source_image="/absolute/path/to/source/image.<ext>",
        target_faces_index=[0],
        source_faces_index=[0],
        input_directory="/absolute/path/to/target/images",
        output_directory="/absolute/path/to/store/results",
        enhancement_options=enhancement_options,
        detection_options=detection_options,
        face_blur_options=face_blur_options,
        face_mask_correction=True,
        face_mask_correction_size=10,
        skip_if_exists=False,
        progressbar=False,
    )


if __name__ == '__main__':
    main()
```
    
## Example for CodeFormer (solves issues with hair - enhances face only if needed!)

```python
import PIL.Image
from reactorlib import EnhancementOptions, DetectionOptions, enhance_image, settings, FaceEnhancementOptions


def main():
    settings.configure(**{
        'DEVICE': 'CUDA',
    })

    image = PIL.Image.open("</path/to/image.ext>")
    enhancement_options = EnhancementOptions(
        face_enhancement_options=FaceEnhancementOptions(
            do_enhancement=True,
            codeformer_visibility=1.0,
            codeformer_weight=0.5,
            restore_face_only=True,  # this does the job
            detection_options=DetectionOptions(
                det_thresh=0.25,
                det_maxnum=0
            )  # this plays role in face mask detection
        )
    )
    result = enhance_image(image, enhancement_options)


if __name__ == '__main__':
    main()
```

Models are downloaded automatically. Approximately 1.5 GB of free space is required.