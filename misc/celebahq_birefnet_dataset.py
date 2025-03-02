import os
import random
from pathlib import Path
from typing import List

from PIL import Image
from tqdm import tqdm

# Input directories: update these paths to your actual directories
INPUT_IMAGE_DIRECTORY = Path("CelebAMask-HQ/CelebA-HQ-img")
INPUT_MASK_DIRECTORY = Path("CelebAMask-HQ/CelebAHairMask-HQ")

# Output directory for the new structure
OUTPUT_DIRECTORY = Path("CelebAMask-HQ/CelebA-HQ-BiRefNet-Hair-Train")

# Dataset names corresponding to the splits
DATASET_NAMES = ["CelebAHQTrain", "CelebAHQTest", "CelebAHQVal"]


def listdir(directory: str, filter_ext: List = None):
    """
    Recursively list files in a directory with optional filtering by file extension.
    """
    if filter_ext is not None:
        filter_ext = set(filter_ext)
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check file extension if filtering is enabled
            if filter_ext is None or Path(file).suffix.lower() in filter_ext:
                file_list.append(os.path.join(root, file))
    return file_list


def main():
    # List all image files (adjust extensions as needed)
    image_paths = listdir(str(INPUT_IMAGE_DIRECTORY), filter_ext=[".jpg", ".png", ".jpeg"])

    # Shuffle for random splitting; fixed seed for reproducibility
    random.seed(42)
    random.shuffle(image_paths)

    total = len(image_paths)
    train_count = int(total * 0.8)
    test_count = int(total * 0.1)
    # Use the remaining images for validation
    val_count = total - train_count - test_count

    print(f"Total images: {total}")
    print(f"Train: {train_count}, Test: {test_count}, Val: {val_count}")

    # Create splits
    splits = {
        DATASET_NAMES[0]: image_paths[:train_count],
        DATASET_NAMES[1]: image_paths[train_count:train_count + test_count],
        DATASET_NAMES[2]: image_paths[train_count + test_count:],
    }

    for dataset_name, paths in splits.items():
        print(f"Processing {dataset_name} with {len(paths)} images...")
        for path in tqdm(paths, desc=f"Processing {dataset_name}"):
            stem = Path(path).stem
            # Construct corresponding mask path; assumes mask filenames match image filenames
            mask_path = INPUT_MASK_DIRECTORY / f"{stem}.png"

            if not mask_path.exists():
                print(f"Warning: No mask found for {path}, skipping.")
                continue

            # Open image and mask
            image = Image.open(path)
            mask_img = Image.open(mask_path)

            # Define destination paths: images under "im", masks under "gt"
            dst_image_path = OUTPUT_DIRECTORY / dataset_name / "im" / f"{stem}.jpg"
            dst_mask_path = OUTPUT_DIRECTORY / dataset_name / "gt" / f"{stem}.png"

            # Ensure destination directories exist
            os.makedirs(dst_image_path.parent, exist_ok=True)
            os.makedirs(dst_mask_path.parent, exist_ok=True)

            # Save files
            image.save(dst_image_path)
            mask_img.save(dst_mask_path)


if __name__ == '__main__':
    main()
