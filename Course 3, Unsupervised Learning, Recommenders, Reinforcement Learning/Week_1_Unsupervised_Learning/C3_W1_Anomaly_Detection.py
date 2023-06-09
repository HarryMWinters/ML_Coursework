"""
Anomaly Detection

- Implement detection of (gaussian) mean and variance of features.
"""
import typing as t

import numpy as np

M = t.TypeVar("M", bound=int)
N = t.TypeVar("N", bound=int)


def estimate_gaussian(
    X: np.ndarray[t.Tuple[M, N], np.dtype[np.float64]]
) -> t.Tuple[
    np.ndarray[t.Tuple[N], np.dtype[np.float64]],
    np.ndarray[t.Tuple[N], np.dtype[np.float64]],
]:
    """
    Calculates mean and variance of all features in the dataset

    Args:
        X (ndarray): (m, n) Data matrix

    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)

    # Equivalent to
    # var = (1 / len(X)) * np.sum((X - mu) ** 2, axis=0)

    return mu, var


def select_threshold(
    y_val: np.ndarray[t.Tuple[M], np.dtype[np.uint8]],
    p_val: np.ndarray[t.Tuple[M], np.dtype[np.uint8]],
) -> t.Tuple[float, float]:
    """
    Finds the best threshold to use for selecting outliers based on the results from a
    validation set (p_val) and the ground truth (y_val)

    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set

    Returns:
        epsilon (float): Threshold chosen
        F1 (float):      F1 score by choosing epsilon as threshold
    """

    best_epsilon = 0
    best_F1 = 0
    F1 = 0

    step_size = (max(p_val) - min(p_val)) / 1000

    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        true_positives = np.sum(np.logical_and(p_val < epsilon, y_val == 1))
        false_positives = np.sum(np.logical_and(p_val < epsilon, y_val == 0))
        false_negatives = np.sum(np.logical_and(p_val >= epsilon, y_val == 1))

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        F1 = (2 * precision * recall) / (precision + recall)

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1
