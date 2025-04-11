import numpy as np
import cv2
import random

def _dist(a, b): # Calculate the distance between 2 points in a 3D coordinate system.
    return (np.sqrt(np.sum( (b-a)**2, 2)))

def apply_k_means(img, k, max_iter):
    # split image into channels, reformat h x w x c structure 
    img = np.array(cv2.split(img))
    img = img.transpose(1, 2, 0)

    # exit conditions
    iter = 0
    moved = True

    # initial cluster centres - randomly initialize
    clusters = [[random.randint(0, 255) for i in range(3)] for j in range(k)]

    iter = 0
    while iter <= max_iter and moved:
        iter += 1

        # calculate distance between pixels and cluster, for every cluster
        distances = [_dist(img, clusters[i]) for i in range(k)]        

        # index (0, ..., k) of the nearest cluster centre for each pixel
        nearest = np.argmin(distances, 0) 

        prev_clusters = clusters.copy()

        for i in range(k):
            # create mask of which pixels belong to the cluster
            ind = np.array(np.where(nearest == i, 1, 0), dtype=bool) 
            
            # Check if any pixels are assigned to this cluster
            if np.any(ind):
                # apply mask to image to extract subset of pixels 
                subset = img[ind] 

                # calculate mean of the identified subset - update cluster centres
                clusters[i] = [
                    np.round(np.mean(subset[:,0])),
                    np.round(np.mean(subset[:,1])),
                    np.round(np.mean(subset[:,2]))
                ]
            # If no pixels assigned, keep the previous cluster center
            # (or you could reinitialize randomly)

        # Check if clusters have moved
        if np.array_equal(clusters, prev_clusters):
            moved = False

    # Convert clusters to integer type
    clusters = np.array(clusters, dtype=int)
    # Create segmented image by mapping each pixel to its cluster color
    img2 = clusters[nearest]
    return img2


# def apply_k_means(img, k, max_iter):
#     # split image into channels, reformat h x w x c structure 
#     img = np.array(cv2.split(img))
#     img = img.transpose(1, 2, 0)

#     # exit conditions
#     iter = 0
#     moved = True

#     # initial cluster centres
#     clusters = [[random.randint(0, 255) for i in range(3)] for j in range(k) ]

#     iter = 0
#     while iter <= max_iter and moved == True:
#         iter += 1

#         # calculate distance between pixels and cluster, for every cluster
#         distances = [_dist(img, clusters[i]) for i in range(k)]        

#         # index (0, ..., k) of the nearest cluster centre for each pixel
#         # produces an array the same shape as the image, instead of pixels,
#         # it stores in the index of the nearest cluster
#         # this can be used as a mask later on
#         nearest = np.argmin(distances, 0) 

#         prev_clusters = clusters.copy()

#         for i in range(k):

#             # create 1-hot encoded mask of which pixels belong to the cluster
#             ind = np.array( np.where(nearest == i, 1, 0), dtype = bool) 

#             # apply mask to image to extract subset of pixels 
#             subset = img[ind] 

#             # calculate mean of the identified subset - update cluster centres
#             clusters[i] = [np.round(np.mean(subset[:,0])),
#                         np.round(np.mean(subset[:,1])),
#                         np.round(np.mean(subset[:,2]))]

#             # remove NaN values - replace with 0
#             if np.isnan(clusters[i][0]):
#                 clusters[i][0] = 0
#             if np.isnan(clusters[i][1]):
#                 clusters[i][1] = 0
#             if np.isnan(clusters[i][2]):
#                 clusters[i][2] = 0

#         if clusters == prev_clusters:
#             moved = False

#     # after the final iteration, the cluster centres represent the pixel colour of each cluster
#     # we apply the final version of the array, nearest, as a mask to sample colours for each pixel

#     clusters = np.array(clusters, dtype = int)
#     img2 = clusters[nearest]
#     return img2

