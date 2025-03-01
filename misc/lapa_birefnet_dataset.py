import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

INPUT_DIRECTORY = Path("dataset/dis/LaPa")
OUTPUT_DIRECTORY = Path("dataset/dis/FaceParsing/LaPa")

DIRECTORY_INDEX = 2
FOLDERS = ["train", "val", "test"]
DATASET_DIRECTORY = ["LaPaTrain", "LaPaVal", "LaPaTest"]

"""
label	class
0	background
1	skin
2	left eyebrow
3	right eyebrow
4	left eye
5	right eye
6	nose
7	upper lip
8	inner mouth
9	lower lip
10	hair
"""

BG_COLOR = 0
FACE_COLOR = 255

LAPA_FACE_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def listdir(directory: str, filter_ext: List = None):
    if filter_ext is not None:
        filter_ext = set(filter_ext)

    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            to_append = None
            if filter_ext is None:
                to_append = os.path.join(root, file)
            else:
                if Path(file).suffix in filter_ext:
                    to_append = os.path.join(root, file)
            if to_append is not None:
                file_list.append(to_append)

    return file_list


def main():
    paths = listdir(str(INPUT_DIRECTORY))
    for path in tqdm(paths):
        is_image_dir = Path(path).parent.name == "images"
        is_dataset_directory = Path(path).parent.parent.name == FOLDERS[DIRECTORY_INDEX]

        if not (is_image_dir and is_dataset_directory):
            continue

        stem = Path(path).stem
        labels_path = Path(path).parent.parent / "labels" / f"{stem}.png"

        image = Image.open(path)
        label_img = Image.open(labels_path)

        dst_image_path = OUTPUT_DIRECTORY / DATASET_DIRECTORY[DIRECTORY_INDEX] / "im" / f"{stem}.jpg"
        dst_mask_path = OUTPUT_DIRECTORY / DATASET_DIRECTORY[DIRECTORY_INDEX] / "gt" / f"{stem}.png"

        label_np = np.array(label_img, dtype=np.uint8)
        binary_mask = np.isin(label_np, LAPA_FACE_INDICES).astype(np.uint8)
        binary_mask_255 = binary_mask * 255
        mask_image = Image.fromarray(binary_mask_255, mode='L')

        # from PIL import ImageOps, ImageFilter
        # blurred_mask = mask_image.filter(ImageFilter.GaussianBlur(radius=2))
        # overlayed_img = image.copy()
        # overlayed_img.putalpha(mask_image)
        # overlayed_img.show()

        if not os.path.isfile(dst_mask_path.parent):
            os.makedirs(dst_mask_path.parent, exist_ok=True)
        if not os.path.isfile(dst_image_path.parent):
            os.makedirs(dst_image_path.parent, exist_ok=True)

        image.save(dst_image_path)
        mask_image.save(dst_mask_path)


if __name__ == '__main__':
    main()
