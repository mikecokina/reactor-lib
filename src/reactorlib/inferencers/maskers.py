from ..conf.settings import FaceMasker, settings

from .. inferencers.birefnet_mask_generator import BiRefNetMaskGenerator
from .. inferencers.bisenet_mask_generator import BiSeNetMaskGenerator

from .. shared import SharedModelKeyMixin, SingletonBase

# todo: unify caches


class MaskerCache(SharedModelKeyMixin, SingletonBase):
    """First singleton that includes the shared model/key properties."""
    pass


class BisenetCache(SharedModelKeyMixin, SingletonBase):
    pass


bisenet_cache = BisenetCache()
masker_cache = MaskerCache()


def get_hair_masker_from_cache() -> BisenetCache:
    if bisenet_cache.model is None:
        bisenet_cache.model = BiSeNetMaskGenerator()
    return bisenet_cache


def get_face_masker_from_cache() -> MaskerCache:
    if (masker_cache.model is None) or (masker_cache.key != settings.FACE_MASKER.value):
        if settings.FACE_MASKER in [FaceMasker.birefnet_L, FaceMasker.birefnet_T]:
            masker_cache.model = BiRefNetMaskGenerator()
        elif settings.FACE_MASKER == FaceMasker.bisenet:
            masker_cache.model = BiSeNetMaskGenerator()
        else:
            raise NotImplementedError(f"{settings.FACE_SWAPPER} not implemented")
        masker_cache.key = settings.FACE_MASKER.value
    return masker_cache
