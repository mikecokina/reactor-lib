from ..conf.settings import FaceMasker, settings

from .. inferencers.birefnet_mask_generator import BiRefNetMaskGenerator
from .. inferencers.bisenet_mask_generator import BiSeNetMaskGenerator

from .. shared import SharedModelKeyMixin, SingletonBase


class MaskerCache(SharedModelKeyMixin, SingletonBase):
    """First singleton that includes the shared model/key properties."""
    pass


masker_cache = MaskerCache()


def get_masker_cache() -> MaskerCache:
    if (masker_cache.model is None) or (masker_cache.key != settings.FACE_MASKER.value):
        if settings.FACE_MASKER == FaceMasker.birefnet:
            masker_cache.model = BiRefNetMaskGenerator()
        elif settings.FACE_MASKER == FaceMasker.bisenet:
            masker_cache.model = BiSeNetMaskGenerator()
        else:
            raise NotImplementedError(f"{settings.FACE_SWAPPER} not implemented")
        masker_cache.key = settings.FACE_MASKER.value
    return masker_cache
