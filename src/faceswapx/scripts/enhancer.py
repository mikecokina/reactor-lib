import os.path
from pathlib import Path
from PIL import Image
from faceswapx.codeformer.codeformer_model import enhance_image

from faceswapx import settings, FaceEnhancementOptions, EnhancementOptions, DetectionOptions

IMAGE_EXTENSIONS = ('.webp', '.png', '.jpg', '.jpeg', '.gif', '.bmp')


def process_images(
        input_dir,
        output_dir,
        codeformer_visibility: float = 1.0,
        codeformer_weight: float = 0.5,
        restore_face_only: bool = True,
        allowed_extensions=IMAGE_EXTENSIONS,
        skip_if_exists: bool = False
):
    # Create the output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define enhancement options
    face_enhancement_options = EnhancementOptions(
            face_enhancement_options=FaceEnhancementOptions(
                do_enhancement=True,
                enhance_target=False,
                codeformer_visibility=codeformer_visibility,
                codeformer_weight=codeformer_weight,
                restore_face_only=restore_face_only,
                face_detection_options=DetectionOptions(
                    det_thresh=0.25,
                    det_maxnum=0
                )
            ),
        )

    # Loop through all files in the input directory
    for image_path in Path(input_dir).glob('*'):
        if image_path.suffix.lower() in allowed_extensions:
            try:
                # Output path
                output_path = output_dir / image_path.name

                if skip_if_exists and os.path.isfile(output_path):
                    continue

                # Open image
                with Image.open(image_path) as input_image:
                    # Enhance image
                    result_image = enhance_image(input_image, face_enhancement_options)

                    # Save the processed image to the output directory with the same filename
                    result_image.save(output_path)

                    print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")


if __name__ == "__main__":
    settings.configure(**{
        'DEVICE': str('cuda').upper(),
    })

    process_images(
        input_dir="",
        output_dir="",
        codeformer_weight=0.5,
        codeformer_visibility=1.0,
        skip_if_exists=True,
        restore_face_only=True
    )
