import os

from PIL import Image
from tqdm import tqdm


def crop(
        folder_path,
        output_folder_path,
        start_x,
        crop_width,
        new_height=1280
):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    # Iterate over all images in the folder
    for filename in tqdm(os.listdir(folder_path)):

        input_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder_path, filename)

        try:
            with Image.open(input_path) as img:
                original_width, original_height = img.size

                original_width, original_height = img.size

                # Ensure the crop bounds are within the image dimensions
                end_x = start_x + crop_width
                if start_x < 0 or end_x > original_width:
                    raise ValueError("Crop region is out of image bounds.")

                cropped_img = img.crop((start_x, 0, end_x, original_height))

                # Calculate the new width to maintain the aspect ratio
                aspect_ratio = crop_width / original_height
                new_width = int(new_height * aspect_ratio)

                # Resize the cropped image
                resized_img = cropped_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resized_img.save(output_path)

                # Save the flipped image to the output directory
                resized_img.save(output_path)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
