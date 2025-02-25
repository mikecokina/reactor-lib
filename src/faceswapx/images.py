from typing import Text

import cv2
import numpy as np
import torch

from PIL import Image, ImageOps

from .upscalers import LANCZOS


def fix_image(image: Image.Image):
    if image is None:
        return None
    # noinspection PyBroadException
    try:
        image = ImageOps.exif_transpose(image)
        image = fix_png_transparency(image)
    except Exception:
        pass
    return image


def fix_png_transparency(image: Image.Image):
    if image.mode not in ("RGB", "P") or not isinstance(image.info.get("transparency"), bytes):
        return image
    image = image.convert("RGBA")
    return image


def read(fp: Text, **kwargs):
    image = Image.open(fp, **kwargs)
    image = fix_image(image)
    return image


def bgr_image_to_rgb_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert a BGR NumPy image in [0..1] range to a PyTorch RGB float32 tensor."""
    assert img.shape[2] == 3, "image must be RGB"
    if img.dtype == "float64":
        img = img.astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.transpose((2, 0, 1))).float()


def rgb_tensor_to_bgr_image(tensor: torch.Tensor, *, min_max=(0.0, 1.0)) -> np.ndarray:
    """
    Convert a PyTorch RGB tensor in range `min_max` to a BGR NumPy image in [0..1] range.
    """
    tensor = tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    assert tensor.dim() == 3, "tensor must be RGB"
    img_np = tensor.numpy().transpose(1, 2, 0)
    if img_np.shape[2] == 1:  # gray image, no RGB/BGR required
        return np.squeeze(img_np, axis=2)
    return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)


def resize_image(resize_mode, im, width, height, upscaler_name=None):
    """
    Resizes an image with the specified resize_mode, width, and height.

    Args:
        resize_mode: The mode to use when resizing the image.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height, maintaining the aspect ratio,
                and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio,
                and then center the image within the dimensions, filling empty with data from image.
        im: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
        upscaler_name: The name of the upscaler to use. If not provided, defaults to opts.upscaler_for_img2img.
    """

    upscaler_name = "UpscalerLanczos"

    def resize(img, w, h):
        if upscaler_name is None or upscaler_name == "None" or img.mode == 'L':
            return img.resize((w, h), resample=LANCZOS)

        if img.width != w or im.height != h:
            img = img.resize((w, h), resample=LANCZOS)
        return img

    if resize_mode == 0:
        res = resize(im, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if fill_height > 0:
                res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if fill_width > 0:
                res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return res


def get_image(
        img: Image.Image | str,
        **kwargs
) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    return read(img, **kwargs)
