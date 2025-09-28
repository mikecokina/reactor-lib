import os
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch.nn
from PIL import Image
from scipy.ndimage import binary_erosion
from torchvision import transforms
from skimage import transform

from ..conf.settings import settings, FaceMaskModels, FaceMasker, HairMasker, HairMaskModels
from ..decorators import cpu_offload
from ..inferencers.mask_generator import BaseMaskGenerator
from ..inferencers.mixins import MaskGeneratorMixin, TorchMaskGeneratorMixin
from ..shared import download_model, resolve_device
from ..u2net.u2net_network import U2NETFull


__all__ = [
    "U2NetFaceMaskGenerator",
    "U2NetHairMaskGenerator"
]


class ToTensorLab:
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, flag: int = 0):
        self.flag = flag

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:

        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        tmp_label = np.zeros(label.shape)

        label = label if np.max(label) < 1e-6 else label / np.max(label)

        tmp_im = np.zeros((image.shape[0], image.shape[1], 3))
        image = image / np.max(image)
        if image.shape[2] == 1:
            tmp_im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmp_im[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmp_im[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmp_im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmp_im[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmp_im[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmp_label[:, :, 0] = label[:, :, 0]

        tmp_im = tmp_im.transpose((2, 0, 1))
        tmp_label = label.transpose((2, 0, 1))

        return {
            'imidx': torch.from_numpy(imidx.copy()).long(),  # or int64
            'image': torch.from_numpy(tmp_im.copy()).float(),  # ensure float
            'label': torch.from_numpy(tmp_label.copy()).float()  # ensure float
        }


class RescaleT:

    def __init__(self, output_size: int):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        img = transform.resize(
            image,
            (self.output_size, self.output_size),
            mode='constant'
        )
        lbl = transform.resize(
            label,
            (self.output_size, self.output_size),
            mode='constant',
            order=0,
            preserve_range=True
        )
        return {'imidx': imidx, 'image': img, 'label': lbl}


class U2NetMaskGenerator(BaseMaskGenerator, MaskGeneratorMixin, TorchMaskGeneratorMixin):
    MASK_MODEL_CONFIG: Union[FaceMaskModels, HairMaskModels] = None

    def __init__(
            self,
            model_type: Union[FaceMasker, HairMasker],
            model_path: Union[str, Path] = None,
    ):
        self._image_size = 1024
        self.device = resolve_device(settings.device, settings.DEVICE_ID)

        masker_options = self.MASK_MODEL_CONFIG.get_config(model_type)

        if model_path and os.path.isfile(model_path):
            self.model_path = model_path
        elif model_path and os.path.isdir(model_path):
            self.model_path = os.path.join(model_path, masker_options.filename)
        else:
            self.model_path = os.path.join(settings.U2NET_MODEL_DIR, masker_options.filename)

        if not os.path.exists(self.model_path):
            download_model(self.model_path, masker_options.url)

        self.model = self.load_net()

    def load_net(self) -> torch.nn.Module:
        net = U2NETFull()
        net.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        net.to('cpu')

        if not settings.CPU_OFFLOAD:
            net.to(self.device)

        net.eval()

        return net

    def name(self) -> str:
        return "U2Net"

    @cpu_offload
    def generate_mask(
            self,
            # image in cv2 BGR form
            face_image: np.ndarray,
            *args,
            **kwargs
    ) -> np.ndarray:
        """
           1) Load the image with PIL.
           2) Apply the same transforms as training (RescaleT, ToTensorLab).
           3) Run a forward pass through U^2-Net.
           4) Return (orig_pil, mask_pil).
           """
        # --- Convert image to RGB form
        face_image = face_image[:, :, ::-1]

        # --- Load the original image
        orig_pil = Image.fromarray(face_image).convert('RGB')
        input_size = orig_pil.size

        # --- Use the same transform as training
        transf = transforms.Compose([
            RescaleT(self._image_size),  # resize shorter side to 320
            ToTensorLab(flag=0)  # convert to tensor in [0..1]
        ])

        # The transforms expect a dict with keys 'imidx', 'image', and 'label'.
        # Provide a dummy imidx as a NumPy array and a dummy 3-channel label.
        sample = {
            'imidx': np.array([0]),  # Now a NumPy array instead of an int
            'image': np.array(orig_pil),  # shape [H, W, 3]
            'label': np.zeros((orig_pil.size[1], orig_pil.size[0], 3), dtype=np.uint8)  # dummy label [H, W, 3]
        }
        sample = transf(sample)
        # Now sample['image'] is a Torch tensor with shape [3, H, W]

        # --- Prepare batch dimension & move to device
        img_tensor = sample['image'].unsqueeze(0).float()  # [1, 3, H, W]
        img_tensor = img_tensor.to(self.device)

        # --- Forward pass
        with torch.no_grad():
            d0, d1, d2, d3, d4, d5, d6 = self.model(img_tensor)

        # --- Take the final output map (d0) and normalize it: shape [1, 1, H, W]
        pred = d0[:, 0, :, :]  # remove the channel dim → [1, H, W]
        pred = self.norm_pred(pred)  # normalize to [0..1]

        # --- Convert to NumPy array scaled to [0..255]
        pred_np = pred.squeeze().cpu().numpy()  # shape [H, W]
        pred_np_255 = (pred_np * 255).astype(np.uint8)
        mask_pil = Image.fromarray(pred_np_255, mode='L').resize(orig_pil.size)
        mask_pil = mask_pil.resize(input_size)

        return np.array(mask_pil, dtype=np.uint8)

    @staticmethod
    def norm_pred(d):
        """Normalize the predicted saliency map (d) to [0,1]."""
        ma = torch.max(d)
        mi = torch.min(d)
        eps = 1e-8  # to avoid division-by-zero
        dn = (d - mi) / (ma - mi + eps)
        return dn

    @staticmethod
    def alpha_matting_cutout(
            img: Image.Image,
            mask: Image.Image,
            foreground_threshold: int,
            background_threshold: int,
            erode_structure_size: int,
    ) -> Image.Image:
        """
        Perform alpha matting on an image using a given mask and threshold values.

        This function takes a PIL image `img` and a PIL image `mask` as input, along with
        the `foreground_threshold` and `background_threshold` values used to determine
        foreground and background pixels. The `erode_structure_size` parameter specifies
        the size of the erosion structure to be applied to the mask.

        The function returns a PIL image representing the cutout of the foreground object
        from the original image.
        """
        from pymatting import estimate_alpha_cf, estimate_foreground_ml, stack_images

        if img.mode == "RGBA" or img.mode == "CMYK":
            img = img.convert("RGB")

        img_array = np.asarray(img)
        mask_array = np.asarray(mask)

        is_foreground = mask_array > foreground_threshold
        is_background = mask_array < background_threshold

        structure = None
        if erode_structure_size > 0:
            structure = np.ones(
                (erode_structure_size, erode_structure_size), dtype=np.uint8
            )

        is_foreground = binary_erosion(is_foreground, structure=structure)
        is_background = binary_erosion(is_background, structure=structure, border_value=1)

        trimap = np.full(mask_array.shape, dtype=np.uint8, fill_value=128)
        trimap[is_foreground] = 255
        trimap[is_background] = 0

        img_normalized = img_array / 255.0
        trimap_normalized = trimap / 255.0

        alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
        foreground = estimate_foreground_ml(img_normalized, alpha)
        cutout = np.array(stack_images(foreground, alpha))

        cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
        cutout = Image.fromarray(cutout)

        return cutout


class U2NetFaceMaskGenerator(U2NetMaskGenerator):
    MASK_MODEL_CONFIG = FaceMaskModels


class U2NetHairMaskGenerator(U2NetMaskGenerator):
    MASK_MODEL_CONFIG = HairMaskModels
