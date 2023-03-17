"""
Implement linear regression with one variable to predict profits for a restaurant 
franchise.
https://www.coursera.org/learn/machine-learning/programming/jsE7w/week-2-practice-lab-linear-regression

Suppose you are the CEO of a restaurant franchise and are considering different cities 
for opening a new outlet.

You would like to expand your business to cities that may give your restaurant higher 
profits. The chain already has restaurants in various cities and you have data for 
profits and populations from the cities. You also have data on cities that are 
candidates for a new restaurant.

For these cities, you have the city population.

Can you use the data to help you identify which cities may potentially yield higher 
profits?

"""

from typing import Tuple, TypeVar

import numpy as np

Shape = TypeVar("Shape")


def compute_cost(
    x: np.ndarray[Shape, np.dtype[np.float64]],
    y: np.ndarray[Shape, np.dtype[np.float64]],
    w: float,
    b: float,
) -> float:
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0]

    # You need to return this variable correctly
    total_cost = 0

    ### START CODE HERE ###
    for city_pop, profit in zip(x, y):
        prediction = w * city_pop + b
        loss = (profit - prediction) ** 2
        total_cost += loss

    total_cost = total_cost / (2 * m)
    ### END CODE HERE ###

    return total_cost


# UNQ_C2
# GRADED FUNCTION: compute_gradient
def compute_gradient(
    x: np.ndarray[Shape, np.dtype[np.float64]],
    y: np.ndarray[Shape, np.dtype[np.float64]],
    w: float,
    b: float,
) -> Tuple[float, float]:
    """
    Computes the gradient for linear regression
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities)
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b
    """

    # Number of training examples
    m = x.shape[0]

    cum_dj_dw = cum_dj_db = 0

    for x_i, y_i in zip(x, y):
        f_wb = (w * x_i) + b

        dj_db = f_wb - y_i
        dj_dw = (f_wb - y_i) * x_i

        cum_dj_dw += dj_dw
        cum_dj_db += dj_db

    cum_dj_dw = cum_dj_dw / m
    cum_dj_db = cum_dj_db / m

    return float(cum_dj_dw), float(cum_dj_db)
