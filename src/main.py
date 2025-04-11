import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import uuid
from image_seg import apply_k_means

# Importing an image
def import_image(filename): 

    img = cv2.imread(filename)
    if img is None:
        raise ValueError(f"Image at {filename} could not be loaded. Please check the file path and format.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # swap colour channels so that it is displayed correctly
    return img

def result(img, no_plot , should_save, save_dpi):
    if(not no_plot):
        # display image
        plt.axis('off')
        plt.imshow(img)

    if(should_save):
        name = str(uuid.uuid4()) + ".png" # use uuid to avoid name collisions while saving
        path = "../output/" + name
        plt.savefig(path, format="png", dpi=save_dpi)

def main():
    parser = argparse.ArgumentParser(description="Sample script")
    
    # Add arguments
    parser.add_argument("-i", "--input", type=str, help='The input image to perform image segmentation on', required=True)
    parser.add_argument("-s", "--save", action="store_true", help="Saves the resulting image to ../output directory if provided", default=False)
    parser.add_argument("-np", "--no-plot", action="store_true", help="prevents plotting the image on the screen", default=False)
    parser.add_argument("-d", "--dpi", type=int, help="With how much dpi the resulting image should be saved with, the default value is 600", required=False, default=600)
    parser.add_argument("-k", "--k-value", type=int, help="The k value to provide as a parameter to k-means clustering", required=True)
    parser.add_argument("-max", "--max-iter", type=int, help="max_iterations to perform if centroids not still optimized in k-means clustering", default=10)

    # Parse arguments
    args = parser.parse_args()

    img = import_image(args.input)
    res = apply_k_means(img, args.k_value, args.max_iter)
    result(res, args.no_plot, args.save, args.dpi)

if __name__ == "__main__":
    main()