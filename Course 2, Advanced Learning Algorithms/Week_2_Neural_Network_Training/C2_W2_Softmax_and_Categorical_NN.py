"""
Course 2, Week 2, Lab 1: Softmax and Categorical NN

- Use keras/tf to build a NN with 3 layers and a linear output layer.
- Manually compute softmax.
"""

import typing as t

import numpy as np
import tensorflow as tf

T = t.TypeVar("T")


def my_softmax(
    z: np.ndarray[T, np.dtype[np.float64]]
) -> np.ndarray[T, np.dtype[np.float64]]:
    """Softmax converts a vector of values to a probability distribution.

    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """
    ### START CODE HERE ###
    denom = np.sum(np.exp(z))
    a = np.exp(z) / denom
    ### END CODE HERE ###
    return a


model = tf.keras.Sequential(
    [
        ### START CODE HERE ###
        tf.keras.layers.InputLayer(input_shape=(400,)),
        tf.keras.layers.Dense(25, activation="relu"),
        tf.keras.layers.Dense(15, activation="relu"),
        tf.keras.layers.Dense(10, activation="linear")
        ### END CODE HERE ###
    ],
    name="my_model",
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

model.build()
