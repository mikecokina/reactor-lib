from abc import abstractmethod

import numpy as np
from PIL import Image

from .conf.settings import ImageUpscalerOptions

# noinspection PyUnresolvedReferences
LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
# noinspection PyUnresolvedReferences
NEAREST = (Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST)


class Upscaler:
    name = None

    def __init__(self, options: ImageUpscalerOptions):
        self.upscaler = None

        self.scale = options.scale
        self.tile = options.tile
        self.tile_pad = options.tile_pad

        self.load_model()

    @abstractmethod
    def upscale(self, image: Image.Image) -> Image.Image:
        return image

    @abstractmethod
    def _validate_scale(self, scale: int) -> bool:
        pass

    @abstractmethod
    def load_model(self):
        pass


class UpscalerNone(Upscaler):
    name = "None"

    def _validate_scale(self, scale: int) -> bool:
        return True

    def load_model(self):
        pass

    def upscale(self, image) -> Image.Image:
        return image


class UpscalerLanczos(Upscaler):
    name = "Lanczos"

    def _validate_scale(self, scale: int) -> bool:
        return True

    def upscale(self, image: Image.Image):
        return image.resize((int(image.width * self.scale), int(image.height * self.scale)), resample=LANCZOS)

    def load_model(self):
        pass


class UpscalerNearest(Upscaler):
    name = "Nearest"

    def _validate_scale(self, scale: int) -> bool:
        return True

    def upscale(self, image: Image.Image):
        image = super().upscale(image)
        return image.resize((int(image.width * self.scale), int(image.height * self.scale)), resample=NEAREST)

    def load_model(self):
        pass


class UpscalerRealESRGAN(Upscaler):
    def __init__(self, options: ImageUpscalerOptions):
        self.pre_pad = 0

        self._validate_scale(options.scale)
        super().__init__(options)

    def load_model(self):
        from spandrel.architectures.RealESRGAN.arch.RRDBNet import RRDBNet
        from .realsergan.realesrgan_model import RealESRGANer

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=self.scale
        )

        upscaler = RealESRGANer(
            scale=self.scale,
            net=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=False
        )
        self.upscaler = upscaler

    def upscale(self, image: Image.Image) -> Image.Image:
        img, mode = self.upscaler.enhance(img=np.array(image, dtype=np.uint8))
        return Image.fromarray(img).convert(mode)

    def _validate_scale(self, scale: int) -> bool:
        if scale not in (2, 4):
            raise ValueError("Scale must be 2 or 4 for RealESRGAN model")
        return True


class UpscalerSwinIRSmall(Upscaler):
    def __init__(self, options: ImageUpscalerOptions):
        self._validate_scale(options.scale)
        super().__init__(options)

    def load_model(self):
        from faceswapx.swinir.swinir_model import SwinIRRealSmall
        self.upscaler = SwinIRRealSmall(tile=self.tile, tile_pad=self.tile_pad)

    def upscale(self, image: Image.Image) -> Image.Image:
        return self.upscaler.upscale_4x(image)

    def _validate_scale(self, scale: int) -> bool:
        # todo: rethink this validators, use name of class like _4x or so?
        # todo: default SwinIR upsacle is hardcoded to 4 anyway
        return True


class UpscalerSwinIRLarge(Upscaler):
    def load_model(self):
        from faceswapx.swinir.swinir_model import SwinIRRealLarge
        self.upscaler = SwinIRRealLarge(tile=self.tile, tile_pad=self.tile_pad,)

    def upscale(self, image: Image.Image) -> Image.Image:
        return self.upscaler.upscale_4x(image)

    def _validate_scale(self, scale: int) -> bool:
        return True
