import random

import cv2
import numpy as np


def _dist(a, b):
    """Calculate the distance between 2 points in a 3D coordinate system."""
    return np.sqrt(np.sum((b - a) ** 2, 2))


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

    while iter_count < max_iter and moved:
        iter_count += 1

        # Calculate distance between pixels and each cluster
        distances = np.array([_dist(img_copy, clusters[i]) for i in range(k)])

        # Index (0, ..., k) of the nearest cluster center for each pixel
        nearest = np.argmin(distances, 0)

        # Store previous clusters for comparison
        prev_clusters = clusters.copy()

        # Update each cluster center
        for i in range(k):
            # Create mask of which pixels belong to the cluster
            ind = np.where(nearest == i, True, False)

            # Check if any pixels are assigned to this cluster
            if np.any(ind):
                # Apply mask to image to extract subset of pixels
                subset = img_copy[ind]

                # Calculate mean of the identified subset - update cluster centers
                # Handle potential empty clusters
                if len(subset) > 0:
                    clusters[i] = [
                        np.round(np.mean(subset[:, 0])),
                        np.round(np.mean(subset[:, 1])),
                        np.round(np.mean(subset[:, 2])),
                    ]
            # If no pixels assigned, keep the previous cluster center

        # Check if clusters have moved significantly
        if np.allclose(clusters, prev_clusters):
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


def apply_hsv_k_means(img, k, max_iter):
    """
    Apply K-means clustering in HSV color space to segment an image.

    Args:
        img: Input image in BGR format (OpenCV default)
        k: Number of clusters
        max_iter: Maximum number of iterations

    Returns:
        Segmented image in HSV format where each pixel is replaced with its cluster color
    """
    # Convert input image to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Make a copy of the image to avoid modifying the original
    img_copy = hsv_img.copy()

    # Split image into channels, reformat to h x w x c structure
    img_copy = np.array(cv2.split(img_copy))
    img_copy = img_copy.transpose(1, 2, 0)

    # Exit conditions
    iter_count = 0
    moved = True

    # Initial cluster centers - randomly initialize with HSV ranges
    # H: 0-179, S: 0-255, V: 0-255 (OpenCV HSV ranges)
    clusters = np.array(
        [
            [
                random.randint(0, 179),  # H
                random.randint(0, 255),  # S
                random.randint(0, 255),  # V
            ]
            for j in range(k)
        ],
        dtype=np.float32
    )

    while iter_count < max_iter and moved:
        iter_count += 1

        # Calculate distance between pixels and each cluster
        distances = np.array([_dist(img_copy, clusters[i]) for i in range(k)])

        # Index (0, ..., k) of the nearest cluster center for each pixel
        nearest = np.argmin(distances, 0)

        # Store previous clusters for comparison
        prev_clusters = clusters.copy()

        # Update each cluster center
        for i in range(k):
            # Create mask of which pixels belong to the cluster
            ind = np.where(nearest == i, True, False)

            # Check if any pixels are assigned to this cluster
            if np.any(ind):
                # Apply mask to image to extract subset of pixels
                subset = img_copy[ind]

                # Calculate mean of the identified subset - update cluster centers
                # Handle potential empty clusters
                if len(subset) > 0:
                    # For HSV, H is circular (0-179), so we need to handle it differently
                    # For now, simple approach: just calculate mean
                    clusters[i] = [
                        np.round(np.mean(subset[:, 0])) % 180,  # Wrap H around 180
                        np.round(np.mean(subset[:, 1])),
                        np.round(np.mean(subset[:, 2])),
                    ]
            # If no pixels assigned, keep the previous cluster center

        # Check if clusters have moved significantly
        if np.allclose(clusters, prev_clusters):
            moved = False

        # Print progress
        if iter_count % 2 == 0 or not moved:
            print(
                f"HSV K-means iteration {iter_count}/{max_iter} - {'Converged' if not moved else 'In progress'}"
            )

    # Convert clusters to integer type for image creation
    clusters = np.array(clusters, dtype=np.uint8)

    # Create segmented image by mapping each pixel to its cluster color
    segmented_hsv = clusters[nearest]

    # Return the segmented image in HSV format
    return segmented_hsv


def display_hsv_result(hsv_img):
    """
    Display an HSV image, keeping it in HSV format.

    Args:
        hsv_img: Image in HSV format

    Returns:
        None (displays the image)
    """
    # Create a copy for display
    display_img = hsv_img.copy()

    # Display the image in HSV format
    cv2.imshow("HSV Segmentation Result", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()