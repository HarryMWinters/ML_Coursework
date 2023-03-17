"""
Course Two, Week 1, Lab 1: Neurons and Layers

- Use keras/tensorflow to build a simple neural network with 3 layers.
- Manually compute an activation from a layer given it's weights. Use numpy matmul.

"""
import typing as t

import numpy as np
import tensorflow as tf

N = t.TypeVar("N")
J = t.TypeVar("J")
I = t.TypeVar("I")

model = tf.keras.Sequential(
    [
        # specify input size
        tf.keras.Input(shape=(400,)),
        ### START CODE HERE ###
        tf.keras.layers.Dense(25, activation="sigmoid", name="L1"),
        tf.keras.layers.Dense(15, activation="sigmoid", name="L2"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="L3"),
        ### END CODE HERE ###
    ],
    name="my_model",
)


def my_dense(
    a_in: np.ndarray[t.Tuple[N,], np.dtype[np.float64]],
    W: np.ndarray[t.Tuple[N, J], np.dtype[np.float64]],
    b: np.ndarray[t.Tuple[J,], np.dtype[np.float64]],
    g: t.Callable[
        [np.ndarray[t.Tuple[J,], np.dtype[np.float64]]],
        np.ndarray[t.Tuple[J,], np.dtype[np.float64]],
    ],
) -> np.ndarray[t.Tuple[J,], np.dtype[np.float64]]:
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units
      g    activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (j,))  : j units
    """
    units = W.shape[1]
    a_out = np.zeros(units)

    ### START CODE HERE ###
    a_out = g(np.matmul(a_in, W) + b)
    ### END CODE HERE ###
    return a_out
