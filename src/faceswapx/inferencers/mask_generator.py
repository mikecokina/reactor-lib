from abc import abstractmethod, ABCMeta

import numpy as np


class BaseMaskGenerator(metaclass=ABCMeta):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def generate_mask(
            self,
            face_image: np.ndarray,
            *args,
            **kwargs
    ) -> np.ndarray:
        pass
