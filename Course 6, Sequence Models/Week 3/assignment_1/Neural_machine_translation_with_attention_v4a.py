import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from babel.dates import format_date
from nmt_utils import load_dataset
from nmt_utils import plot_attention_map
from nmt_utils import preprocess_data
from nmt_utils import softmax
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dot
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation="tanh")
densor2 = Dense(1, activation="relu")
activator = Activation(
    softmax, name="attention_weights"
)  # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes=1)


def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product
    of the attention weights "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of
        shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attention) LSTM cell
    """
    s_prev = repeator(s_prev)
    concat = concatenator([a, s_prev])
    e = densor1(concat)
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas, a])

    return context


# number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
n_a = 32
# number of units for the post-attention LSTM's hidden state "s"
n_s = 64

# Please note, this is the post attention LSTM cell.
# Please do not modify this global variable.
post_activation_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(len(machine_vocab), activation=softmax)


def modelf(
    Tx,
    Ty,
    n_a,
    n_s,
    human_vocab_size,
    machine_vocab_size,
):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """

    # Define the inputs of your model with a shape (Tx, human_vocab_size)
    # Define s0 (initial hidden state) and c0 (initial cell state)
    # for the decoder LSTM with shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    # initial hidden state
    s0 = Input(shape=(n_s,), name="s0")
    # initial cell state
    c0 = Input(shape=(n_s,), name="c0")
    # hidden state
    s = s0
    # cell state
    c = c0

    # Initialize empty list of outputs
    outputs = []

    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)
    for _ in range(Ty):
        context = one_step_attention(a, s)
        _, s, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        out = output_layer(s)
        outputs.append(out)

    model = Model(inputs=[X, s0, c0], outputs=outputs)

    return model


opt = Adam(
    lr=0.005,
    beta_1=0.9,
    beta_2=0.999,
    decay=0.01,
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=opt,
    metrics=["accuracy"],
)
