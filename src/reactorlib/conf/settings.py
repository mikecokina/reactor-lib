import json
import os
import os.path as op
from dataclasses import dataclass, field

from logging import config as log_conf


@dataclass
class DetectionOptions:
    det_thresh: float = 0.5
    det_maxnum: int = 0


@dataclass
class ImageEnhancementOptions:
    do_enhancement: bool = True

    scale: int = 2
    tile: int = 400,
    tile_pad: int = 10,
    pre_pad: int = 0,
    half: bool = False


@dataclass
class FaceEnhancementOptions:
    do_enhancement: bool = True
    enhance_target: bool = False
    scale: int = 1
    codeformer_visibility: float = 0.5
    codeformer_weight: float = 0.5
    restore_face_only: bool = False


@dataclass
class EnhancementOptions:
    do_enhancement: bool = True
    enhance_target: bool = False
    codeformer_visibility: float = 0.5
    codeformer_weight: float = 0.5
    restore_face_only: bool = False

    face_enhancement_options: FaceEnhancementOptions = field(default_factory=FaceEnhancementOptions)
    detection_options: DetectionOptions = field(default_factory=DetectionOptions)
    image_enhancement_options: ImageEnhancementOptions = field(default_factory=ImageEnhancementOptions)


@dataclass
class FaceBlurOptions:
    do_face_blur: bool = False
    do_video_noise: bool = False
    blur_radius: int = 3,
    blur_strength: float = 0.5
    noise_pixel_size: int = 1
    noise_strength: float = 0.1
    seed: float = 0
    mask_size: int = 0


class _Const(object):
    FACE_SWAP_MODEL_DOWNLOAD_NAME: str = "inswapper_128.onnx"
    FACE_SWAP_MODEL_URL: str = "https://github.com/facefusion/facefusion-assets/releases" \
                               "/download/models/inswapper_128.onnx"
    FACE_RESTORATION_MODEL: str = "CodeFormer"
    FACE_RESTORATION_MODEL_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    FACE_RESTORATION_MODEL_DOWNLOAD_NAME = "codeformer-v0.1.0.pth"

    IMAGE_RESTORATION_MODEL: str = "RealESRGAN"
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth"
    IMAGE_RESTORATION_MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/" \
                                  "releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    IMAGE_RESTORATION_MODEL_DOWNLOAD_NAME = "RealESRGAN_x2.pth"

    COLORS = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
        (128, 0, 0),
        (0, 128, 0),
        (128, 128, 0),
        (0, 0, 128),
        (0, 128, 128),
    ]

    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png']


class DefaultSettings(object):
    LOG_CONFIG: str = op.join(op.dirname(op.abspath(__file__)), 'logging_schemas/default.json')
    SUPPRESS_LOGGER: bool = False
    DEVICE: str = 'cpu'
    DEVICE_ID: int | str = None
    MODELS_PATH: str = os.path.join(
        op.dirname(op.dirname(op.dirname(op.abspath(__file__)))), "data", "models"
    )
    NO_HALF: bool = True
    FACE_RESTORATION_MODEL_DIR: str = os.path.join(MODELS_PATH, 'codeformer')
    IMAGE_RESTORATION_MODEL_DIR: str = os.path.join(MODELS_PATH, 'realesrgan')
    PROVIDERS = ["CPUExecutionProvider"]


class Settings(_Const, DefaultSettings):
    _instance = None

    # defaults #########################################################################################################
    DEFAULT_SETTINGS = {}
    ####################################################################################################################

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls.DEFAULT_SETTINGS = cls.settings_serializer()
            # Put any initialization here.

        return cls._instance

    @classmethod
    def settings_serializer(cls):
        return {
            "LOG_CONFIG": cls.LOG_CONFIG,
            "SUPPRESS_LOGGER": cls.SUPPRESS_LOGGER,
            "DEVICE": cls.DEVICE,
            "DEVICE_ID": cls.DEVICE_ID,
            "MODELS_PATH": cls.MODELS_PATH,
            "NO_HALF": cls.NO_HALF,
            "PROVIDERS": cls.PROVIDERS,
        }

    @staticmethod
    def load_conf(path):
        with open(path) as f:
            conf_dict = json.loads(f.read())
        log_conf.dictConfig(conf_dict)

    @classmethod
    def set_up_logging(cls):
        if op.isfile(cls.LOG_CONFIG):
            cls.load_conf(cls.LOG_CONFIG)

    @classmethod
    def configure(cls, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(cls, key):
                raise ValueError("You are about to set configuration which doesn't exist")
            setattr(cls, key, value)

            if key == 'LOG_CONFIG':
                cls.set_up_logging()
            if key == 'DEVICE':
                setattr(cls, key, str(value).lower())

                if cls.DEVICE == "cuda":
                    cls.PROVIDERS = ["CUDAExecutionProvider"]
                else:
                    cls.PROVIDERS = ["CPUExecutionProvider"]
            if key == 'MODELS_PATH':
                cls.FACE_RESTORATION_MODEL_DIR = os.path.join(cls.MODELS_PATH, 'codeformer')
                cls.IMAGE_RESTORATION_MODEL_DIR = os.path.join(cls.MODELS_PATH, 'realesrgan')

    @property
    def device(self):
        return self.DEVICE


settings = Settings()
