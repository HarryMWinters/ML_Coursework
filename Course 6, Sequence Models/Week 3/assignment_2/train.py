import glob
import io
import os
import random
import sys

import IPython
import numpy as np
from pydub import AudioSegment
from td_utils import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

Tx = 5511  # The number of time steps input to the model from the spectrogram
n_freq = (
    101  # Number of frequencies input to the model at each time step of the spectrogram
)

Ty = 1375  # The number of time steps in the output of our model

X = np.load("./XY_train/X0.npy")
Y = np.load("./XY_train/Y0.npy")

X = np.concatenate((X, np.load("./XY_train/X1.npy")), axis=0)
Y = np.concatenate((Y, np.load("./XY_train/Y1.npy")), axis=0)

Y = np.swapaxes(Y, 1, 2)

# Load preprocessed dev set examples
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")

# GRADED FUNCTION: model


def modelf(input_shape):
    """
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    X_input = Input(shape=input_shape)

    ### START CODE HERE ###

    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(196, kernel_size=15, strides=4)(X_input)  # CONV1D
    X = BatchNormalization()(X)  # Batch normalization
    X = Activation("relu")(X)  # ReLu activation
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(units=128, return_sequences=True)(
        X
    )  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization

    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units=128, return_sequences=True)(
        X
    )  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 4: Time-distributed dense layer (≈1 line)
    X = TimeDistributed(Dense(1, activation="sigmoid"))(
        X
    )  # time distributed  (sigmoid)

    ### END CODE HERE ###

    model = Model(inputs=X_input, outputs=X)

    return model


model = modelf(input_shape=(Tx, n_freq))
model.summary()

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


model.fit(X, Y, batch_size=20, epochs=100)

loss, acc = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy = ", acc)

from tensorflow.keras.models import model_from_json

json_file = open("./models/model_new3.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("./models/model_new3.h5")
