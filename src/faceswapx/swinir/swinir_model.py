import abc
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..conf.settings import settings, ImageUpscaler, ImageUpsaclerModels
from .network_swinir import SwinIR as Net
from ..shared import torch_gc, DummyTqdm, resolve_device, download_model


class SwinIRModel:
    def __init__(
            self,
            tile: int = 0,
            tile_pad: int = 32,
            progressbar: bool = False,
    ):

        self.model_file = None
        self.device = resolve_device(settings.device, settings.DEVICE_ID)
        self.model = None
        self.param_key_g = None
        self.window_size = None
        self.scale = None

        self.tile = tile
        self.tile_pad = tile_pad

        self._tqdm = tqdm if progressbar else DummyTqdm

    def _upscale(self, image: Image.Image) -> Image.Image:
        torch_gc()

        image = np.array(image, dtype=np.float32)[:, :, ::-1] / 255.

        image = np.transpose(
            image
            if image.shape[2] == 1
            else image[:, :, [2, 1, 0]],
            (2, 0, 1)
        )  # HCW-BGR to CHW-RGB

        image = torch.from_numpy(image).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = image.size()
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
            image = torch.cat([image, torch.flip(image, [2])], 2)[:, :, :h_old + h_pad, :]
            image = torch.cat([image, torch.flip(image, [3])], 3)[:, :, :, :w_old + w_pad]
            output = self.tile_process(image) if self.tile > 0 else self.process(image)
            output = output[..., :h_old * self.scale, :w_old * self.scale]

        # get image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

        return Image.fromarray(output[:, :, ::-1])

    def tile_process(self, image: torch.Tensor) -> torch.Tensor:
        torch_gc()

        # test the image tile by tile
        b, c, h, w = image.size()
        tile = min(self.tile, h, w)
        assert tile % self.window_size == 0, "tile size should be a multiple of window_size"
        sf = self.scale

        stride = tile - self.tile_pad
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        e_ = torch.zeros(b, c, h * sf, w * sf).type_as(image)
        w_ = torch.zeros_like(e_)

        with self._tqdm(
                total=len(h_idx_list) * len(w_idx_list),
                desc="Image Upscaling",
                unit="tiles"
        ) as pbar:
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = image[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    out_patch = self.model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    e_[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
                    w_[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)

                    pbar.update()

        output = e_.div_(w_)

        return output

    def process(self, image: torch.Tensor):
        return self.model(image)


class SwinIRReal(SwinIRModel):
    MODEL = None

    def __init__(
            self,
            tile: int = 0,
            tile_pad: int = 32,
            progressbar: bool = False,
    ):
        super().__init__(
            progressbar=progressbar,
            tile=tile,
            tile_pad=tile_pad,
        )

        # Ensure files are presented
        model_config = ImageUpsaclerModels.get_config(self.MODEL)
        self.model_file = os.path.join(settings.SWINIR_MODEL_DIR, model_config.filename)

        if not os.path.isfile(self.model_file):
            download_model(self.model_file, model_config.url)

        self.window_size = 8
        self.scale = 4
        self.param_key_g = 'params_ema'

        model = self._get_network()
        pretrained_model = torch.load(self.model_file)
        model.load_state_dict(
            pretrained_model[self.param_key_g]
            if self.param_key_g in pretrained_model.keys()
            else pretrained_model,
            strict=True
        )

        model = model.eval()
        self.model = model.to(self.device)

    @abc.abstractmethod
    def _get_network(self) -> Net:
        pass

    def upscale_4x(self, image: Image.Image) -> Image.Image:
        return self._upscale(image)


class SwinIRRealSmall(SwinIRReal):
    MODEL = ImageUpscaler.swinir_real_sr

    def _get_network(self) -> Net:
        return Net(
            upscale=self.scale,
            in_chans=3,
            img_size=64,
            window_size=self.window_size,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='nearest+conv',
            resi_connection='1conv'
        )


class SwinIRRealLarge(SwinIRReal):
    MODEL = ImageUpscaler.swinir_real_sr_large

    def _get_network(self) -> Net:
        return Net(
            upscale=self.scale,
            in_chans=3,
            img_size=64,
            window_size=self.window_size,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
            embed_dim=240,
            num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
            mlp_ratio=2,
            upsampler='nearest+conv',
            resi_connection='3conv'
        )
