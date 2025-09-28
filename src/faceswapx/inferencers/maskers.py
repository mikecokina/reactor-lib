from .u2net_mask_generator import U2NetFaceMaskGenerator, U2NetHairMaskGenerator
from ..conf.settings import FaceMasker, settings, HairMasker
from ..inferencers.birefnet_mask_generator import BiRefFaceNetMaskGenerator
from ..inferencers.bisenet_mask_generator import BiSeNetMaskGenerator
from ..shared import SharedModelKeyMixin, SingletonBase


# todo: unify caches


class FaceMaskerCache(SharedModelKeyMixin, SingletonBase):
    """First singleton that includes the shared model/key properties."""
    ...


class HairMaskerCache(SharedModelKeyMixin, SingletonBase):
    ...


hair_masker_cache = HairMaskerCache()
face_masker_cache = FaceMaskerCache()


def get_hair_masker_from_cache() -> HairMaskerCache:
    if (hair_masker_cache.model is None) or (hair_masker_cache.key != settings.HAIR_MASKER.value):
        if settings.HAIR_MASKER == HairMasker.bisenet:
            hair_masker_cache.model = BiSeNetMaskGenerator()
        elif settings.HAIR_MASKER == HairMasker.u2net:
            hair_masker_cache.model = U2NetHairMaskGenerator(model_type=HairMasker.u2net)
        else:
            raise NotImplementedError(f"{settings.HAIR_MASKER} not implemented")

    return hair_masker_cache


def get_face_masker_from_cache() -> FaceMaskerCache:
    if (face_masker_cache.model is None) or (face_masker_cache.key != settings.FACE_MASKER.value):
        if settings.FACE_MASKER == FaceMasker.birefnet_large:
            face_masker_cache.model = BiRefFaceNetMaskGenerator(model_type=FaceMasker.birefnet_large)
        elif settings.FACE_MASKER == FaceMasker.birefnet_tiny:
            face_masker_cache.model = BiRefFaceNetMaskGenerator(model_type=FaceMasker.birefnet_tiny)
        elif settings.FACE_MASKER == FaceMasker.bisenet:
            face_masker_cache.model = BiSeNetMaskGenerator()
        elif settings.FACE_MASKER == FaceMasker.u2net:
            face_masker_cache.model = U2NetFaceMaskGenerator(model_type=FaceMasker.u2net)
        else:
            raise NotImplementedError(f"{settings.FACE_SWAPPER} not implemented")
        face_masker_cache.key = settings.FACE_MASKER.value
    return face_masker_cache
