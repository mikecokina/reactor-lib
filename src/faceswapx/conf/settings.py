import json
import os
import os.path as op
from dataclasses import dataclass, field
from enum import Enum

from logging import config as log_conf
from pathlib import Path


@dataclass
class DetectionOptions:
    det_thresh: float = 0.5
    det_maxnum: int = 0
    mask_size: int = 0
    mask_blur_kernel: int = 12
    mask_from_source: bool = False
    reverse_detection_order: bool = False


@dataclass
class ImageUpscalerOptions:
    scale: int = 2
    tile: int = 400
    tile_pad: int = 10
    pre_pad: int = 0
    half: bool = False


@dataclass
class FaceEnhancementOptions:
    do_enhancement: bool = True
    enhance_target: bool = False
    enhance_loops: int = 1
    scale: int = 1

    codeformer_visibility: float = 0.5
    codeformer_weight: float = 0.5

    restore_face_only: bool = False
    paste_back_hair: bool = False

    face_size: int = 512

    face_detection_options: DetectionOptions = field(default_factory=DetectionOptions)
    hair_detection_options: DetectionOptions = field(default_factory=DetectionOptions)


@dataclass
class EnhancementOptions:
    face_enhancement_options: FaceEnhancementOptions = field(default_factory=FaceEnhancementOptions)
    image_enhancement_options: ImageUpscalerOptions = field(default_factory=ImageUpscalerOptions)


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


class FaceSwapper(Enum):
    inswapper_128 = 'inswapper_128'
    inswapper_256 = 'inswapper_256'
    inswapper_512 = 'inswapper_512'
    reswapper_128 = 'reswapper_128'
    reswapper_256 = 'reswapper_256'
    reswapper_256_1567500 = 'reswapper_256_1567500'


class FaceMasker(Enum):
    bisenet = "bisenet"
    birefnet_large = "birefnet_large"
    birefnet_tiny = "birefnet_tiny"


class HairMasker(Enum):
    bisenet = "bisenet"


class ImageUpscaler(Enum):
    realesrgan_2x = "realesrgan_2x"
    realesrgan_4x = "realesrgan_4x"
    swinir_real_sr = "swinir_real_sr"
    swinir_real_sr_large = "swinir_real_sr_large"


@dataclass
class ModelOptions:
    filename: str
    url: str


class FaceMaskModels(object):
    _config = {
        FaceMasker.birefnet_large.value: ModelOptions(
            filename="birefnet-lapa-face-epoch_20--swin-v1-large.onnx",
            url="https://huggingface.co/mikestealth/birefnet/resolve/main/"
                "birefnet-lapa-face-epoch_20--swin-v1-large.onnx"
        ),
        FaceMasker.birefnet_tiny.value: ModelOptions(
            filename="birefnet-lapa-face-epoch_20--swin-v1-tiny.onnx",
            url="https://huggingface.co/mikestealth/birefnet/resolve/main/"
                "birefnet-lapa-face-epoch_20--swin-v1-tiny.onnx"
        )
    }

    @classmethod
    def get_config(cls, face_masker: FaceMasker) -> ModelOptions:
        if face_masker == FaceMasker.bisenet:
            raise NotImplementedError("Such combination is not allowed!")

        if face_masker.value in cls._config:
            return cls._config[face_masker.value]
        raise AttributeError(f"No such model {face_masker.value}")


class ImageUpsaclerModels(object):
    _config = {
        ImageUpscaler.realesrgan_2x.value: ModelOptions(
            filename="RealESRGAN_x2.pth",
            url="https://huggingface.co/mikestealth/RealESRGAN/resolve/main/RealESRGAN_x2.pth"
        ),
        ImageUpscaler.realesrgan_4x.value: ModelOptions(
            filename="RealESRGAN_x4.pth",
            url="https://huggingface.co/mikestealth/RealESRGAN/resolve/main/RealESRGAN_x4.pth"
        ),
        ImageUpscaler.swinir_real_sr.value: ModelOptions(
            filename="003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
            url="https://huggingface.co/mikestealth/SwinIR/resolve/main/"
                "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth?download=true"
        ),
        ImageUpscaler.swinir_real_sr_large.value: ModelOptions(
            filename="003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
            url="https://huggingface.co/mikestealth/SwinIR/resolve/main/"
                "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
        ),

    }

    @classmethod
    def get_config(cls, upscaler: ImageUpscaler) -> ModelOptions:
        if upscaler.value in cls._config:
            return cls._config[upscaler.value]
        raise AttributeError(f"No such upscaler {upscaler.value}")


class FaceSwapperModels(object):
    _config = {
        FaceSwapper.inswapper_128.value: ModelOptions(
            filename="inswapper_128.onnx",
            url="https://huggingface.co/mikestealth/inswapper/resolve/main/inswapper_128.onnx"
        ),
        FaceSwapper.inswapper_256.value: ModelOptions(
            filename="inswapper_128.onnx",
            url="https://huggingface.co/mikestealth/inswapper/resolve/main/inswapper_128.onnx"
        ),
        FaceSwapper.inswapper_512.value: ModelOptions(
            filename="inswapper_128.onnx",
            url="https://huggingface.co/mikestealth/inswapper/resolve/main/inswapper_128.onnx"
        ),
        FaceSwapper.reswapper_128.value: ModelOptions(
            filename="reswapper_128-1019500-newarch.pth",
            url="https://huggingface.co/mikestealth/reswapper/resolve/main/reswapper_128-1019500-newarch.pth"
        ),
        FaceSwapper.reswapper_256.value: ModelOptions(
            filename="reswapper_256-1399500-newarch.pth",
            url="https://huggingface.co/mikestealth/reswapper/resolve/main/reswapper_256-1399500-newarch.pth"
        ),
        FaceSwapper.reswapper_256_1567500.value: ModelOptions(
            filename="reswapper_256-1567500-newarch.pth",
            url="https://huggingface.co/mikestealth/reswapper/resolve/main/reswapper_256-1567500-newarch.pth"
        )
    }

    @classmethod
    def get_config(cls, face_swapper: FaceSwapper) -> ModelOptions:
        if face_swapper.value in cls._config:
            return cls._config[face_swapper.value]
        raise AttributeError(f"No such model {face_swapper.value}")


class _Const(object):
    DISABLE_NSFW = True
    FACE_MASKER = FaceMasker.bisenet
    FACE_SWAPPER = FaceSwapper.inswapper_128
    _default_model = FaceSwapperModels.get_config(FaceSwapper.inswapper_128)

    FACE_SWAPPER_MODEL_DOWNLOAD_NAME: str = _default_model.filename
    FACE_SWAPPER_MODEL_URL: str = _default_model.url

    FACE_RESTORATION_MODEL: str = "CodeFormer"
    FACE_RESTORATION_MODEL_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    FACE_RESTORATION_MODEL_DOWNLOAD_NAME = "codeformer-v0.1.0.pth"

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
    CPU_OFFLOAD = False

    MODELS_PATH: str = str(Path(op.abspath(__file__)).parent.parent.parent / "data" / "models")
    NO_HALF: bool = True
    FACE_RESTORATION_MODEL_DIR: str = os.path.join(MODELS_PATH, 'codeformer')
    REALESRGAN_MODEL_DIR: str = os.path.join(MODELS_PATH, 'realesrgan')
    SWINIR_MODEL_DIR: str = os.path.join(MODELS_PATH, 'swinir')
    BIREFNET_MODEL_DIR: str = os.path.join(MODELS_PATH, 'birefnet')

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
            "CPU_OFFLOAD": cls.CPU_OFFLOAD,
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

                cls.PROVIDERS = ["CPUExecutionProvider"]
                if cls.DEVICE == "cuda":
                    cls.PROVIDERS = ["CUDAExecutionProvider"]

            if key == 'MODELS_PATH':
                cls.FACE_RESTORATION_MODEL_DIR = os.path.join(cls.MODELS_PATH, 'codeformer')
                cls.BIREFNET_MODEL_DIR = os.path.join(cls.MODELS_PATH, 'birefnet')
                cls.SWINIR_MODEL_DIR: str = os.path.join(cls.MODELS_PATH, 'swinir')
                cls.REALESRGAN_MODEL_DIR: str = os.path.join(cls.MODELS_PATH, 'realesrgan')

            if key == 'FACE_SWAPPER':
                if not isinstance(value, FaceSwapper):
                    raise ValueError("FACE_SWAPPER value have to be a value of FaceSwapper enum")
                cls.FACE_SWAPPER = value
                model_config: ModelOptions = FaceSwapperModels.get_config(value)
                cls.FACE_SWAPPER_MODEL_DOWNLOAD_NAME = model_config.filename
                cls.FACE_SWAPPER_MODEL_URL = model_config.url

    @property
    def device(self):
        return self.DEVICE


settings = Settings()
