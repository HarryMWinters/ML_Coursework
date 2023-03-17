"""
Model Evaluation: Bias, Variance, and Regularization

- Code mean squared error (MSE) function. 
- Code categorization error (CERR) function.
- Use l2 regularization to reduce overfitting.

"""
import typing as t

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense  # pylint: disable=import-error

# pylint: disable-next=import-error
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential  # pylint: disable=import-error
from tensorflow.keras.optimizers import Adam  # pylint: disable=import-error

M = t.TypeVar("M", bound=t.Tuple)


# UNQ_C1
# GRADED CELL: eval_mse
def eval_mse(
    y: np.ndarray[M, np.dtype[np.float64]],
    yhat: np.ndarray[M, np.dtype[np.float64]],
) -> np.floating:
    """
    Calculate the mean squared error on a data set.
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:
      err: (scalar)
    """
    err = (1 / (2 * len(y))) * (np.sum((y - yhat) ** 2))

    ### END CODE HERE ###

    return err


# UNQ_C2
# GRADED CELL: eval_cat_err
def eval_cat_err(
    y: np.ndarray[M, np.dtype[np.float64]],
    yhat: np.ndarray[M, np.dtype[np.float64]],
) -> np.floating:
    """
    Calculate the categorization error
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:|
      cerr: (scalar)
    """
    m = len(y)
    incorrects = y != yhat
    cerr = np.sum(incorrects) / m

    return cerr


# UNQ_C3
# GRADED CELL: model

tf.random.set_seed(1234)
model = keras.models.Sequential(
    [
        Dense(120, activation="relu", name="L0"),
        Dense(40, activation="relu", name="L1"),
        Dense(6, activation="linear", name="L2"),
    ],
    name="Complex",
)
model.compile(
    ### START CODE HERE ###
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=0.01),
    ### END CODE HERE ###
)

# UNQ_C4
# GRADED CELL: model_s

tf.random.set_seed(1234)
model_s = Sequential(
    [
        ### START CODE HERE ###
        Dense(6, activation="relu", name="L1"),
        Dense(6, activation="linear", name="L2")
        ### END CODE HERE ###
    ],
    name="Simple",
)
model_s.compile(
    ### START CODE HERE ###
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=0.01),
    ### START CODE HERE ###
)

# UNQ_C5
# GRADED CELL: model_r

tf.random.set_seed(1234)
model_r = Sequential(
    [
        ### START CODE HERE ###
        Dense(
            120,
            activation="relu",
            name="L0",
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
        ),
        Dense(
            40,
            activation="relu",
            name="L1",
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
        ),
        Dense(6, activation="linear", name="L2"),
        ### START CODE HERE ###
    ],
    name=None,
)
model_r.compile(
    ### START CODE HERE ###
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=0.01),
    ### START CODE HERE ###
)
