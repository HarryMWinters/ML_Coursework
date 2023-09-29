import math

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
import tensorflow.keras.layers as tfl
from cnn_utils import *
from matplotlib.pyplot import imread
from PIL import Image
from tensorflow.python.framework import ops
from test_utils import comparator
from test_utils import summary

# GRADED FUNCTION: happyModel


def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE

    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes.
    Normally, functions should take these values as function parameters.

    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """
    model = tf.keras.Sequential(
        [
            tfl.ZeroPadding2D(padding=3, input_shape=(64, 64, 3)),
            tfl.Conv2D(filters=32, kernel_size=(7, 7), strides=1),
            tfl.BatchNormalization(axis=3),
            tfl.ReLU(),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(units=1, activation="sigmoid"),
        ]
    )

    return model


# GRADED FUNCTION: convolutional_model


def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE

    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes.
    Normally, functions should take these values as function parameters.

    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """

    input_img = tf.keras.Input(shape=input_shape)

    Z1 = tfl.Conv2D(filters=8, kernel_size=(4, 4), strides=1, padding="SAME")(input_img)
    A1 = tfl.ReLU()(Z1)
    P1 = tfl.MaxPool2D(pool_size=(8, 8), strides=8, padding="SAME")(A1)
    Z2 = tfl.Conv2D(filters=16, kernel_size=(2, 2), strides=1, padding="SAME")(P1)
    A2 = tfl.ReLU()(Z2)
    P2 = tfl.MaxPool2D(pool_size=(4, 4), strides=4, padding="SAME")(A2)
    F = tfl.Flatten()(P2)

    outputs = tfl.Dense(units=6, activation="softmax")(F)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model
