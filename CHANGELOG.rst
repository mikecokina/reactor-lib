Change Log
==========
|


v0.1.dev0_
-----
.. v0.1.dev0_: https://github.com/mikecokina/reactor-lib

**Release date:** 2024-##-##

**Features**
    - Image Upsaclers
        - RealESRGAN x2
        - RealESRGAN x4
        - SwinIR Real x4
        - SwinIR Real Large x4

* **Supported**
    - Image-to-Image Face Swap
        - Single source face to multiple different targets
        - Single source face to single target
        - Multiple source faces to multiple targets defined by index mapping
        - Bulk swapping from single image as source to images within target directory
            - supported file extensions `.jpg`, `.jpeg`, `.png`
    - Image-to-Video Face Swap
    - Image Enhancement
        - Enhancements of swapped faces with CodeFormer
        - Enhancements of swapped faces with GFPGAN
        - Supports face enhancements only
            - Provided via segmentation of human faces via Bisenet or Birefnet (trained by myself)
    - Suported Face Swap Models
        - InSwappper 128 (fake implementation for 256 and 512 by pixel-shifting)
        - ReSwapper 128 and 256


* **Not Supported**
    - Model-based source (will not be implemented)
