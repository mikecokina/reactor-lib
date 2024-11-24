__version__ = '0.1.dev0'

from .conf.settings import settings
from .react import swap

from .codeformer.codeformer_model import enhance_image
from .codeformer.codeformer_model import enhance_image as enhance_face
from .realsergan.realesrgan_model import enhance_image as enhance_image

from .conf.settings import EnhancementOptions, DetectionOptions, FaceBlurOptions, ImageEnhancementOptions
