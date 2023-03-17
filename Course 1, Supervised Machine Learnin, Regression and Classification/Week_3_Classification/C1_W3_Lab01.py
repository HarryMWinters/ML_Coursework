"""
Implement logistic regression with two variables to predict college admittance.
https://www.coursera.org/learn/machine-learning/programming/vPfeK/week-3-practice-lab-logistic-regression/lab?path=%2Fnotebooks%2FC1_W3_Logistic_Regression.ipynb
"""

import math
from typing import Any, Tuple, TypeVar

import numpy as np

M = TypeVar("M")
N = TypeVar("N")


def sigmoid(
    z: np.ndarray[M, np.dtype[np.float64]],
) -> np.ndarray[M, np.dtype[np.float64]]:
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """

    ### START CODE HERE ###
    g = 1 / (1 + (math.e**-z))
    ### END SOLUTION ###

    return g


def compute_cost(
    X: np.ndarray[Tuple[M, N], np.dtype[np.float64]],
    y: np.ndarray[M, np.dtype[np.float64]],
    w: np.ndarray[N, np.dtype[np.float64]],
    b: float,
    *unused_argv: Any
) -> float:
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below

    Returns:
      total_cost : (scalar) cost
    """

    m, _ = X.shape

    ### START CODE HERE ###
    f_wb = np.dot(X, w) + b
    g_wb = sigmoid

    cost = (-y * np.log(g_wb(f_wb))) - ((1 - y) * np.log(1 - g_wb(f_wb)))
    total_cost = np.sum(cost) / m

    ### END CODE HERE ###

    return float(total_cost)


def compute_gradient(
    X: np.ndarray[Tuple[M, N], np.dtype[np.float64]],
    y: np.ndarray[M, np.dtype[np.float64]],
    w: np.ndarray[N, np.dtype[np.float64]],
    b: float,
    *unused_argv: Any
) -> Tuple[float, np.ndarray[N, np.dtype[np.float64]]]:
    """
    Computes the gradient for logistic regression

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w.
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b.
    """
    m, _ = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.0

    ### START CODE HERE ###

    f_wb = sigmoid(np.dot(X, w) + b)

    dj_dw = (1 / m) * np.sum((f_wb - y)[:, np.newaxis] * X, axis=0)

    dj_db += (1 / m) * np.sum((f_wb - y))

    ### END CODE HERE ###

    return dj_db, dj_dw


def predict(
    X: np.ndarray[Tuple[M, N], np.dtype[np.float64]],
    w: np.ndarray[N, np.dtype[np.float64]],
    b: float,
) -> np.ndarray[M, np.dtype[np.bool_]]:
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m, _ = X.shape
    p = np.zeros(m)

    ### START CODE HERE ###

    f_wb = sigmoid(np.dot(X, w) + b)
    p = f_wb > 0.5

    ### END CODE HERE ###
    return p


def compute_cost_reg(
    X: np.ndarray[Tuple[M, N], np.dtype[np.float64]],
    y: np.ndarray[M, np.dtype[np.float64]],
    w: np.ndarray[N, np.dtype[np.float64]],
    b: float,
    lambda_: float = 1.0,
) -> float:
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar, float) Controls amount of regularization
    Returns:
      total_cost : (scalar)     cost
    """

    m, _ = X.shape

    # Calls the compute_cost function that you implemented above
    cost_without_reg = compute_cost(X, y, w, b)

    # You need to calculate this value
    reg_cost = 0.0

    ### START CODE HERE ###

    reg_cost = (lambda_ / (2 * m)) * np.sum(w**2)

    ### END CODE HERE ###

    # Add the regularization cost to get the total cost
    total_cost = cost_without_reg + reg_cost

    return float(total_cost)


def compute_gradient_reg(
    X: np.ndarray[Tuple[M, N], np.dtype[np.float64]],
    y: np.ndarray[M, np.dtype[np.float64]],
    w: np.ndarray[N, np.dtype[np.float64]],
    b: float,
    lambda_: float = 1.0,
) -> Tuple[float, np.ndarray[N, np.dtype[np.float64]]]:
    """
    Computes the gradient for logistic regression with regularization

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar,float)  regularization constant
    Returns
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b.
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w.

    """
    m, _ = X.shape

    dj_db, dj_dw = compute_gradient(X, y, w, b)

    ### START CODE HERE ###

    dj_dw += (lambda_ / m) * w

    ### END CODE HERE ###

    return dj_db, dj_dw
