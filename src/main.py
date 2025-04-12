import argparse
import os
import sys
import uuid

import cv2
import matplotlib.pyplot as plt

from image_segmentation import apply_k_means


def import_image(filename):
    """
    Load an image from a file and convert it to RGB format.

    Args:
        filename (str): Path to the image file

    Returns:
        numpy.ndarray: The loaded image in RGB format

    Raises:
        ValueError: If the image cannot be loaded
    """
    img = cv2.imread(filename)
    if img is None:
        raise ValueError(
            f"Image at {filename} could not be loaded. Please check the file path and format."
        )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display
    return img


def result(img, no_plot=False, should_save=False, save_dpi=300, output_dir="../output"):
    """
    Display and/or save the image result.

    Args:
        img (numpy.ndarray): The image to display or save
        no_plot (bool): If True, skip displaying the image
        should_save (bool): If True, save the image to a file
        save_dpi (int): DPI for the saved image
        output_dir (str): Directory to save the image in

    Returns:
        str or None: Path to the saved image if saved, otherwise None
    """
    save_path = None

    if not no_plot:
        plt.figure(figsize=(10, 8))
        plt.axis("off")
        plt.imshow(img)
        plt.tight_layout()

    if should_save:
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Generate filename and path
            filename = f"{uuid.uuid4()}.png"
            save_path = os.path.join(output_dir, filename)

            # Save the figure
            plt.savefig(save_path, format="png", dpi=save_dpi, bbox_inches="tight")
            print(f"Image saved to {save_path}")
        except Exception as e:
            print(f"Error saving image: {e}")

    if not no_plot:
        plt.show()

    return save_path


def main():
    # Create argument parser with more descriptive help
    parser = argparse.ArgumentParser(
        description="Image segmentation using K-means clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show default values in help
    )

    # Add arguments with more descriptive help messages
    parser.add_argument(
        "-i", "--input", type=str, help="Path to the input image file", required=True
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save the resulting image to the output directory",
        default=False,
    )
    parser.add_argument(
        "-np",
        "--no-plot",
        action="store_true",
        help="Skip displaying the image on screen",
        default=False,
    )
    parser.add_argument(
        "-d", "--dpi", type=int, help="DPI resolution for the saved image", default=600
    )
    parser.add_argument(
        "-k",
        "--k-value",
        type=int,
        help="Number of clusters for K-means clustering",
        required=True,
    )
    parser.add_argument(
        "-max",
        "--max-iter",
        type=int,
        help="Maximum iterations for K-means clustering convergence",
        default=10,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Directory to save output images",
        default="../output",
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    # Validate k-value is positive
    if args.k_value <= 0:
        print("Error: k-value must be positive.")
        sys.exit(1)

    try:
        # Import the image
        img = import_image(args.input)

        # Apply K-means
        print(f"Applying K-means with k={args.k_value}, max_iter={args.max_iter}...")
        res = apply_k_means(img, args.k_value, args.max_iter)

        # Display/save result
        saved_path = result(res, args.no_plot, args.save, args.dpi, args.output_dir)

        if saved_path:
            print(f"Success! Image saved to: {saved_path}")
        elif not args.no_plot:
            print("Image displayed. Close the window to exit.")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
