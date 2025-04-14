import argparse
import os
import sys
import uuid

import cv2
import matplotlib.pyplot as plt

from image_segmentation import apply_k_means, apply_hsv_k_means, display_hsv_result


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
    save_path = None

    # Always render image to figure, even if not shown
    plt.figure(figsize=(10, 8))
    plt.axis("off")
    plt.imshow(img)
    plt.tight_layout()

    if should_save:
        try:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{uuid.uuid4()}.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, format="png", dpi=save_dpi, bbox_inches="tight", pad_inches=0)
            print(f"Image saved to {save_path}")
        except Exception as e:
            print(f"Error saving image: {e}")

    if not no_plot:
        plt.show()
    else:
        plt.close()  # close the figure if we’re not showing it

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
    parser.add_argument(
        "-c",
        "--color-space",
        type=str,
        choices=["rgb", "hsv"],
        help="Color space to use for segmentation (rgb or hsv)",
        default="rgb",
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

        # Apply K-means depending on color space
        if args.color_space == "rgb":
            print(f"Applying RGB K-means with k={args.k_value}, max_iter={args.max_iter}...")
            res = apply_k_means(img, args.k_value, args.max_iter)

        elif args.color_space == "hsv":
            print(f"Applying HSV K-means with k={args.k_value}, max_iter={args.max_iter}...")
            res = apply_hsv_k_means(img, args.k_value, args.max_iter)

            # HSV sonucu doğrudan HSV formatında göstermek istersen:
            if not args.no_plot:
                display_hsv_result(res)

            # HSV'yi RGB'ye çevirip kaydetmek istersen (opsiyonel):
            res = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)


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
