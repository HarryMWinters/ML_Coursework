"""
k-Means Clustering Algorithm 

- Implement finding closest centroid.
- Implement centroid assignment.
"""

import typing as t

import numpy as np

M = t.TypeVar("M", bound=int)
N = t.TypeVar("N", bound=int)
_K = t.TypeVar("_K", bound=int)


def find_closest_centroids(
    X: np.ndarray[t.Tuple[M, N], np.dtype[np.uint32]],
    centroids: np.ndarray[t.Tuple[_K, N], np.dtype[np.uint32]],
) -> np.ndarray[t.Tuple[M,], np.dtype[np.uint32]]:
    """
    Computes the centroid memberships for every example

    Args:
        X (ndarray): (m, n) Input values
        centroids (ndarray): (K, n) centroids

    Returns:
        idx (array_like): (m,) closest centroids

    """

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    ### START CODE HERE ###
    for i, point in enumerate(X):
        # distances = ||centroid - point||
        distance = np.linalg.norm(centroids - point, axis=1)

        # min_distance = min(distances)
        min_distance_idx = np.argmin(distance)

        # idx[i] = index of min_distance
        idx[i] = min_distance_idx

    ### END CODE HERE ###

    return idx


def compute_centroids(
    X: np.ndarray[t.Tuple[M, N], np.dtype[np.uint32]],
    idx: np.ndarray[t.Tuple[M,], np.dtype[np.uint32]],
    K: _K,
) -> np.ndarray[t.Tuple[_K, N], np.dtype[np.float64]]:
    """
    Returns the new centroids by computing the means of the
    data points assigned to each centroid.

    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each
                       example in X. Concretely, idx[i] contains the index of
                       the centroid closest to example i
        K (int):       number of centroids

    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """

    # Useful variables
    _, n = X.shape

    # You need to return the following variables correctly
    centroids = np.zeros((K, n))

    ### START CODE HERE ###
    for centroid in range(K):
        assigned_points = X[idx == centroid]
        centroids[centroid] = np.mean(assigned_points, axis=0)

    ### END CODE HERE ##

    return centroids
