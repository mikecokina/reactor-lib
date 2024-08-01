[![GitHub version](https://img.shields.io/badge/version-0.1.dev0-yellow.svg)](https://github.com/mikecokina/reactor-lib)
[![Licence GPLv2](https://img.shields.io/badge/license-GNU/GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
[![Python version](https://img.shields.io/badge/python-3.10|3.11-orange.svg)](https://www.python.org/doc/versions/)
[![OS](https://img.shields.io/badge/os-Linux-magenta.svg)](https://www.gnu.org/gnu/linux-and-gnu.html)


# reactor-lib

The face swap python library based on sd-webui-reactor extension 
[setup.py](setup.py)
*Credit goes to*: https://github.com/Gourieff/sd-webui-reactor

## Install

`pip install git+https://github.com/mikecokina/reactor-lib.git@master`


## Example for single image

```python
from reactorlib import settings, swap, DetectionOptions, EnhancementOptions


def main():
    settings.configure(**{
        'DEVICE': 'CUDA',
        # "DEVICE_ID": 0,
        # "MODELS_PATH": "/absolute/path/to/models/directory",
        # "NO_HALF": True,
        # "FACE_RESTORATION_MODEL_DIR": "/absolute/path/to/facerestoration/model/directory",
        # "PROVIDERS": ["CUDAExecutionProvider"],
    })

    enhancement_options = EnhancementOptions(
        upscale_visibility=0.5,
        restorer_visibility=1.0,
        codeformer_weight=0.5,
    )

    detection_options = DetectionOptions(det_thresh=0.5, det_maxnum=0)

    result_image, n_swapped = swap(
        source_image="/absolute/path/to/source/image.<ext>",
        target_image="/absolute/path/to/target/image.<ext>",
        target_faces_index=[0],
        source_faces_index=[0],
        enhancement_options=enhancement_options,
        detection_options=detection_options
    )


if __name__ == "__main__":
    main()
```


## Example for batch processing from single source image

```python
from reactorlib import settings, swap, DetectionOptions, EnhancementOptions


def main():
    settings.configure(**{
        'DEVICE': 'CUDA',
    })

    enhancement_options = EnhancementOptions(
        upscale_visibility=0.5,
        restorer_visibility=1.0,
        codeformer_weight=0.5,
    )

    detection_options = DetectionOptions(det_thresh=0.5, det_maxnum=0)

    _, n_swapped = swap(
        source_image="/absolute/path/to/source/image.<ext>",
        target_faces_index=[0],
        source_faces_index=[0],
        input_directory="/absolute/path/to/target/images",
        output_directory="/absolute/path/to/store/results",
        enhancement_options=enhancement_options,
        detection_options=detection_options
    )


if __name__ == "__main__":
    main()
```

Models are downloaded automatically. Approximately 1.5 GB of free space is required.