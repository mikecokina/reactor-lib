# Patch old and latest version of basicsr used in GFPGAN
import sys
import types
import torchvision.transforms.functional as func


def _apply():
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        # Create a fake module for backwards compatibility
        functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
        functional_tensor.rgb_to_grayscale = func.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor
