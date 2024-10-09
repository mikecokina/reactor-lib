import argparse
from pathlib import Path
from PIL import Image
from reactorlib.codeformer.codeformer_model import enhance_image

from reactorlib import EnhancementOptions, settings

IMAGE_EXTENSIONS = ('.webp', '.png', '.jpg', '.jpeg', '.gif', '.bmp')


def process_images(input_dir, output_dir, codeformer_weight=0.95, allowed_extensions=IMAGE_EXTENSIONS):
    # Create the output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define enhancement options
    enhancement_options = EnhancementOptions(
        upscale_visibility=0.5,
        restorer_visibility=1.0,
        codeformer_weight=codeformer_weight,
    )

    # Loop through all files in the input directory
    for image_path in Path(input_dir).glob('*'):
        if image_path.suffix.lower() in allowed_extensions:
            try:
                # Open image
                input_image = Image.open(image_path)

                # Enhance image
                result_image = enhance_image(input_image, enhancement_options=enhancement_options)

                # Save the processed image to the output directory with the same filename
                output_path = output_dir / image_path.name
                result_image.save(output_path)

                print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance images from a directory and save to a new location.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing images.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory to save enhanced images.')
    parser.add_argument('--codeformer_weight', type=float, default=0.95,
                        help='Weight for the codeformer enhancement (e.g., 0.95).')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for enhancement (e.g., cpu, cuda).')
    parser.add_argument('--model', type=str, default=None,
                        help='Model path to use for the enhancement. '
                             'Requires path up to directory where codeformer/model.pth is stored ')

    args = parser.parse_args()

    settings.configure(**{
        'DEVICE': str(args.device).upper(),
        **({"MODELS_PATH": args.model} if args.model is not None else {})

    })

    process_images(args.input_dir, args.output_dir)
