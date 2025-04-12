import random

import cv2
import numpy as np


def _dist(a, b):
    """Calculate the distance between 2 points in a 3D coordinate system."""
    return np.sqrt(np.sum((b - a) ** 2, 2))

    """
        If pixel RGB = [100, 150, 200] and cluster center = [120, 160, 210],
        then distance = sqrt((20)^2 + (10)^2 + (10)^2) = sqrt(500) ≈ 22.36
    """


"""
1.Selects K random colors as initial cluster centers
2.Assigns each pixel to the nearest cluster
3.Updates each cluster center to be the average color of all its pixels
4.Repeats steps 2-3 until convergence or max iterations
"""

def apply_k_means(img, k, max_iter):
    """
    Apply K-means clustering to segment an image.

    Args:
        img: Input image in BGR or RGB format
        k: Number of clusters
        max_iter: Maximum number of iterations

    Returns:
        Segmented image where each pixel is replaced with its cluster color
    """
    # Make a copy of the image to avoid modifying the original
    img_copy = img.copy()

    # Split image into channels, reformat to h x w x c structure
    img_copy = np.array(cv2.split(img_copy))
    img_copy = img_copy.transpose(1, 2, 0)

    # Exit conditions
    iter_count = 0
    moved = True

    # Initial cluster centers - randomly initialize
    # Ensure clusters are numpy arrays for consistent types
    clusters = np.array(
        [[random.randint(0, 255) for i in range(3)] for j in range(k)], dtype=np.float32
    )
    """
        Ex: if k=3
        Initializes an array such as:
        [[0-255, 0-255, 0-255]
        [0-255, 0-255, 0-255]
        [0-255, 0-255, 0-255]]
    """

    while iter_count < max_iter and moved:
        iter_count += 1

        # Calculate distance between pixels and each cluster
        distances = np.array([_dist(img_copy, clusters[i]) for i in range(k)])
        # Produces an array in the shape: (k, height, width)
        """
            Ex: 2x2 image and k=3
            [
                [
                    [10.5, 15.2],   # Distances from all pixels to cluster 0
                    [8.7, 20.3]
                ],
                
                [
                    [5.3, 12.1],    # Distances from all pixels to cluster 1
                    [9.8, 4.5]
                ],
                
                [
                    [18.9, 7.6],    # Distances from all pixels to cluster 2
                    [11.2, 15.7]
                ]
            ]
]
        """

        # Index (0, ..., k) of the nearest cluster center for each pixel
        nearest = np.argmin(distances, 0)

        """
            Checks each pixel's distance in each cluster and selects the minimum returns it's index
            Ex (above array):
            [
                [1, 2],
                [0, 1]
            ]

            Shape = (height, width)
        """

        # Store previous clusters for comparison
        prev_clusters = clusters.copy()

        # Update each cluster center
        for i in range(k):
            # Create mask of which pixels belong to the cluster
            ind = np.where(nearest == i, True, False)

            """
                i represents a particular cluster in each iteration i = 0, i = 1, ... , i = k-1
                np.where performs a condition check on each element of nearest array, places True where condition is met False otherwise
                
                Ex: (above nearest array) and iteration 1 (i = 1, checking for 2nd cluster) 
                ind = 
                [
                    [True, False],
                    [False, True]
                ]
            """

            # Check if any pixels are assigned to this cluster
            if np.any(ind):
                # Apply mask to image to extract subset of pixels
                subset = img_copy[ind]
                """
                    Eliminate the pixels that are marked as False in the ind array
                    if img_copy is a 2x2 array like this:
                    [
                        [[20,255,10],[10,100,102]],
                        [[5, 102, 154],[25, 244,6]]
                    ]
                    After the above mask is applied:
                    [
                        [[20,255,10]],
                        [[25, 244,6]]
                    ]
                """

                # Calculate mean of the identified subset - update cluster centers
                # Handle potential empty clusters
                if len(subset) > 0:
                    clusters[i] = [
                        np.round(np.mean(subset[:, 0])),
                        np.round(np.mean(subset[:, 1])),
                        np.round(np.mean(subset[:, 2])),
                    ]
                # subset[:, 0] selects the first channel (R: red) of all the pixels in the subset.

                # subset[:, 1] selects the second channel (G: green) of all the pixels.

                # subset[:, 2] selects the third channel (B: blue) of all the pixels.

                """
                    Example on the above array:
                    np.round(np.mean(subset[:, 0])) -> mean(20, 25) = (20 + 25) / 2 = 22.5 -> np.round(22.5) = 22.0
                    np.round performs "banker's rounding", meaning it rounds to the nearest integer but if the decimal is half (.5) then it will round to the even number's side.

                    performs this operation for each color channel on each line.
                    the resulting array for the above example is:

                    clusters[i] = 
                    [
                        22.0, 250.0, 8.0
                    ]
                    thus the ith clusters' position is updated with the mean of each pixel's position that belongs to it.
                """

            # If no pixels assigned, keep the previous cluster center

        # Check if clusters have moved significantly
        if np.allclose(clusters, prev_clusters):
            """
                atol: The absolute tolerance parameter. This defines the maximum allowable difference between a and b for them to be considered equal. Default is 1e-9.
                rtol: The relative tolerance parameter. This defines the maximum allowable difference between a and b relative to the values of a and b themselves. Default is 1e-5.

                The condition for two values a and b to be considered "close" is:
                abs(a−b)≤ atol+rtol×abs(b)
                abs(a−b)≤ atol+rtol×abs(b)
            """
            moved = False

        # Print progress
        if iter_count % 2 == 0 or not moved:
            print(
                f"K-means iteration {iter_count}/{max_iter} - {'Converged' if not moved else 'In progress'}"
            )

    # Convert clusters to integer type for image creation
    clusters = np.array(clusters, dtype=np.uint8)

    # Create segmented image by mapping each pixel to its cluster color
    segmented_img = clusters[nearest]

    return segmented_img
