__version__ = '0.1.dev0'

import os

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

from .patch import numpy_lstsq, retinaface  # noqa: E402

from .conf.settings import settings  # noqa: E402
from .react import swap, image_swap, video_swap  # noqa: E402
# noqa: E402
from .codeformer.codeformer_model import enhance_image as enhance_face  # noqa: E402
from .realsergan.realesrgan_model import enhance_image as enhance_image  # noqa: E402

from .conf.settings import (
    EnhancementOptions,
    DetectionOptions,
    FaceBlurOptions,
    ImageUpscalerOptions,
    FaceEnhancementOptions,
    FaceSwapper,
    FaceMasker
)  # noqa: E402
