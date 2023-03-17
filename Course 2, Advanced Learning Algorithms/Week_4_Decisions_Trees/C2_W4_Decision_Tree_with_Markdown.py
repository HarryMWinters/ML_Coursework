"""
Decision Trees

- Compute the entropy of a node.
- Split a data set into to subsets based on a feature.
- Compute the information gain of a split.
- Compute the highest information feature/split.
"""
import typing as t

import numpy as np

N = t.TypeVar("N")
M = t.TypeVar("M")


def compute_entropy(y: np.ndarray[N, np.dtype[np.uint8]]) -> float:
    """
    Computes the entropy for

    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)

    Returns:
        entropy (float): Entropy at that node

    """

    ### START CODE HERE ###
    # 𝐻(𝑝1)=−𝑝1log2(𝑝1)−(1−𝑝1)log2(1−𝑝1)

    p_1 = np.sum(y) / len(y)

    if p_1 in (0, 1) or len(y) == 0:
        return 0

    Hp_1 = -p_1 * np.log2(p_1) - (1 - p_1) * np.log2(1 - p_1)
    ### END CODE HERE ###

    return Hp_1


def split_dataset(
    X: np.ndarray[t.Tuple[M, N], np.dtype[np.uint8]],
    node_indices: t.List[int],
    feature: int,
) -> t.Tuple[t.List[int], t.List[int]]:
    """
    Splits the data at the given node into left and right branches

    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples
            being considered at this step.
        feature (int):           Index of feature to split on

    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
    """

    # You need to return the following variables correctly
    left_indices = []
    right_indices = []

    ### START CODE HERE ###
    left_indices = [index for index in node_indices if X[index, feature] == 1]
    right_indices = [index for index in node_indices if X[index, feature] == 0]
    ### END CODE HERE ###

    return left_indices, right_indices


def compute_information_gain(
    X: np.ndarray[t.Tuple[M, N], np.dtype[np.uint8]],
    y: np.ndarray[t.Tuple[M], np.dtype[np.uint8]],
    node_indices: t.List[int],
    feature: int,
) -> float:
    """
    Compute the information of splitting the node on a given feature

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target
            variable
        node_indices (ndarray): List containing the active indices. I.e, the samples
            being considered in this step.

    Returns:
        cost (float):        Cost computed

    """
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)

    # Some useful variables
    y_node = y[node_indices]
    y_left = y[left_indices]
    y_right = y[right_indices]

    ### START CODE HERE ###
    entropy_parent = compute_entropy(y_node)
    entropy_left = compute_entropy(y_left)
    entropy_right = compute_entropy(y_right)

    information_gain = (
        entropy_parent
        - (len(y_left) / len(y_node)) * entropy_left
        - (len(y_right) / len(y_node)) * entropy_right
    )
    ### END CODE HERE ###

    return information_gain


def get_best_split(
    X: np.ndarray[t.Tuple[M, N], np.dtype[np.uint8]],
    y: np.ndarray[t.Tuple[M], np.dtype[np.uint8]],
    node_indices: t.List[int],
):
    """
    Returns the optimal feature and threshold value
    to split the node data

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target
            variable
        node_indices (ndarray): List containing the active indices. I.e, the samples
            being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """

    # Some useful variables
    num_features = X.shape[1]

    # You need to return the following variables correctly
    best_feature = -1
    best_information_gain = 0.0

    ### START CODE HERE ###
    for feature in range(num_features):
        information_gain = compute_information_gain(X, y, node_indices, feature)

        if information_gain > best_information_gain:
            best_feature = feature
            best_information_gain = information_gain
    ### END CODE HERE ##

    return best_feature
