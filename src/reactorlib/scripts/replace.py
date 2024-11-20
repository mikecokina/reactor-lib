import os
from PIL import Image, ImageFilter, ImageDraw


def replace_and_blur(
    source_image_path,
    folder_path,
    output_folder_path,
    x1,
    y1,
    x2,
    y2,
    blur_extension=20,
    blur_radius=10,
):
    """
    Replace a bounding box area from the source image into other images, and blur the area
    including an extended boundary around the replaced area.

    :param source_image_path: Path to the source image (e.g., "001.png")
    :param folder_path: Path to the folder containing the target images
    :param output_folder_path: Path to save the modified images
    :param x1: Top-left x-coordinate of the bounding box
    :param y1: Top-left y-coordinate of the bounding box
    :param x2: Bottom-right x-coordinate of the bounding box
    :param y2: Bottom-right y-coordinate of the bounding box
    :param blur_extension: Number of pixels outside the bounding box to blur
    :param blur_radius: Radius of the Gaussian blur applied
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    # Open the source image and crop the specified bounding box
    source_image = Image.open(source_image_path)
    cropped_area = source_image.crop((x1, y1, x2, y2))

    # Iterate over all images in the folder
    for filename in os.listdir(folder_path):
        if filename == os.path.basename(source_image_path):
            continue  # Skip the source image itself

        input_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder_path, filename)

        try:
            with Image.open(input_path) as img:
                # Step 1: Replace content within bounding box
                img_copy = img.copy()
                img_copy.paste(cropped_area, (x1, y1))

                # Step 2: Create a blurred boundary area
                result_img = img_copy.copy()

                # Define extended boundary for blur
                extended_x1 = max(0, x1 - blur_extension)
                extended_y1 = max(0, y1 - blur_extension)
                extended_x2 = min(img.width, x2 + blur_extension)
                extended_y2 = min(img.height, y2 + blur_extension)

                # Create a mask for the blur
                mask = Image.new("L", img.size, 0)
                draw = ImageDraw.Draw(mask)
                # noinspection PyTypeChecker
                draw.rectangle(
                    [(extended_x1, extended_y1), (extended_x2, extended_y2)], fill=255
                )

                # Apply Gaussian blur to the extended region
                blurred_img = result_img.filter(ImageFilter.GaussianBlur(blur_radius))
                result_img.paste(blurred_img, mask=mask)

                # Save the final image
                result_img.save(output_path)
                print(f"Updated {filename} -> {output_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
