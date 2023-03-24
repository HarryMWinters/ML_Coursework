"""
Collaborative Filtering Recommender Systems

This uses the movie lens small dataset, see
https://grouplens.org/datasets/movielens/latest/.

Implements:
 - 

Dataset:
    𝐗 -  (num_users x num_movies matrix) of movie features.
    𝐖 - (N x num_users matrix) of user parameters.
    𝐛 - (num_users x 1 vector) bias vector for users.
    𝑌 - (num_user x num_movies matrix) of user ratings of all movies.
    𝑅 - (num_user x num_movie matrix) binary-valued indicator , where 𝑅(𝑖, 𝑗) = 1 if 
        user 𝑖 gave a rating to movie 𝑗, and 𝑅(𝑖, 𝑗) = 0 otherwise.

"""

# Question 0, Does the length of each feature vector always need to equal the number of
#   users?
# No. But each user weight vector must be the same length as the each movie,s feature
#   vector. Think of this as a aggregation of linear regressions models.

import typing as t

import numpy as np
import tensorflow as tf

N_users = t.TypeVar("N_users", int, np.int32, np.int64)
N_movies = t.TypeVar("N_movies", int, np.int32, np.int64)
N_features = t.TypeVar("N_features", int, np.int32, np.int64)


def cofi_cost_func(
    X: np.ndarray[t.Tuple[N_movies, N_features], np.dtype[np.float64]],
    W: np.ndarray[t.Tuple[N_users, N_features], np.dtype[np.float64]],
    b: np.ndarray[t.Tuple[t.Literal[1], N_features], np.dtype[np.float64]],
    Y: np.ndarray[t.Tuple[N_movies, N_users], np.dtype[np.float64]],
    R: np.ndarray[t.Tuple[N_movies, N_users], np.dtype[np.uint8]],
    lambda_: float,
) -> float:
    """
    Returns the cost for the content-based filtering

    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th
        movies was rated by the j-th user
      lambda_ (float): regularization parameter

    Returns:
      J (float) : Cost
    """
    J = 0

    # Calculate regularization cost
    regularization_cost = lambda_ / 2 * (np.sum(W**2) + np.sum(X**2))
    J += regularization_cost

    # Calculate cost from the error
    # For each movie
    # TODO: This getting tricking. Refactor.
    for movie_idx, (movie_targets, movie_features) in enumerate(zip(Y, X)):
        # For each user
        for (user_idx, user_weights), user_bias, user_target in zip(
            enumerate(W), b[0], movie_targets
        ):
            # Skip calculation if user did not rate movie
            if R[movie_idx, user_idx] == 0:
                continue

            # Calculate cost from the error for each user-move pair
            J += (1 / 2) * (
                np.dot(user_weights, movie_features) + user_bias - user_target
            ) ** 2

    return float(J)


# TODO: Understand this
def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom
        training loop.

    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th
        movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_ / 2) * (
        tf.reduce_sum(X**2) + tf.reduce_sum(W**2)
    )
    return J
