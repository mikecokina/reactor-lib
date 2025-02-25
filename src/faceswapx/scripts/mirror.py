import os

from PIL import Image
from tqdm import tqdm


def mirror(
        folder_path,
        output_folder_path
):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    # Iterate over all images in the folder
    for filename in tqdm(os.listdir(folder_path)):

        input_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder_path, filename)

        try:
            with Image.open(input_path) as img:
                # Flip the image vertically
                flipped_img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

                # Save the flipped image to the output directory
                flipped_img.save(output_path)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
