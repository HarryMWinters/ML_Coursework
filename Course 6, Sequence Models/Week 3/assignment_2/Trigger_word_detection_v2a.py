import typing as t

t.TYPE_CHECKING = True

import numpy as np
from td_utils import graph_spectrogram
from td_utils import match_target_amplitude
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model


def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    # Make sure segment doesn't run past the 10sec background
    segment_start = np.random.randint(low=0, high=10000 - segment_ms)
    segment_end = segment_start + segment_ms - 1

    return (segment_start, segment_end)


def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.

    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the
        existing segments

    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """

    segment_start, segment_end = segment_time

    overlap = False

    for previous_start, previous_end in previous_segments:
        if previous_end >= segment_start and previous_start <= segment_end:
            overlap = True
            break

    return overlap


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step,
    ensuring that the audio segment does not overlap with existing segments.

    Arguments:
    background -- a 10 second background audio recording.
    audio_clip -- the audio clip to be inserted/overlaid.
    previous_segments -- times where audio segments have already been placed

    Returns:
    new_background -- the updated background audio
    """

    segment_ms = len(audio_clip)
    segment_time = get_random_time_segment(segment_ms)

    retry = 5
    while (
        is_overlapping(
            segment_time=segment_time,
            previous_segments=previous_segments,
        )
        and retry >= 0
    ):
        segment_time = get_random_time_segment(segment_ms)
        retry -= 1

    if not is_overlapping(segment_time, previous_segments):
        previous_segments.append(segment_time)
        new_background = background.overlay(audio_clip, position=segment_time[0])

    else:
        new_background = background
        segment_time = (10000, 10000)

    return new_background, segment_time


def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end
    of the segment should be set to 1. By strictly we mean that the label of
    segment_end_y should be 0 while, the 50 following labels should be ones.


    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms

    Returns:
    y -- updated labels
    """
    _, Ty = y.shape
    segment_end_y = int(segment_end_ms * Ty / 10000.0)

    if segment_end_y < Ty:
        for i in range(segment_end_y + 1, segment_end_y + 51):
            if i < Ty:
                y[0, i] = 1

    return y


def create_training_example(background, activates, negatives, Ty):
    """
    Creates a training example with a given background, activates, and negatives.

    Arguments:
        background -- a 10 second background audio recording
        activates -- a list of audio segments of the word "activate"
        negatives -- a list of audio segments of random words that are not "activate"
        Ty -- The number of time steps in the output

    Returns:
        x -- the spectrogram of the training example
        y -- the label at each time step of the spectrogram
    """

    background = background - 20

    y = np.zeros((1, Ty))
    previous_segments = []

    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]

    for one_random_activate in random_activates:
        background, segment_time = insert_audio_clip(
            background,
            one_random_activate,
            previous_segments,
        )

        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end)

    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    for random_negative in random_negatives:
        background, _ = insert_audio_clip(
            background,
            random_negative,
            previous_segments,
        )

    background = match_target_amplitude(background, -20.0)

    file_handle = background.export(
        "train" + ".wav",
        format="wav",
    )

    x = graph_spectrogram("train.wav")

    return x, y


def modelf(input_shape):
    """
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    # Input
    X_input = Input(shape=input_shape)

    # Conv1D
    X = Conv1D(
        filters=196,
        kernel_size=15,
        strides=4,
    )(X_input)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Dropout(rate=0.8)(X)

    # GRU Number 1
    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(rate=0.8)(X)
    X = BatchNormalization()(X)

    # GRU Number 2
    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(rate=0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(rate=0.8)(X)

    # Time-distributed dense layer
    X = TimeDistributed(
        Dense(units=1, activation="sigmoid"),
    )(X)

    model = Model(inputs=X_input, outputs=X)

    return model
