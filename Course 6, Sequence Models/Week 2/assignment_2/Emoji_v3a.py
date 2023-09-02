import typing as t

import numpy as np
from emo_utils import convert_to_one_hot
from emo_utils import predict
from emo_utils import softmax
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def sentence_to_avg(
    sentence: str,
    word_to_vec_map: dict[str, t.Any],
) -> np.ndarray:
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe
    representation of each word and averages its value into a single vector encoding
    the meaning of the sentence.

    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its
        50-dimensional vector representation

    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of
        shape (J,), where J can be any number
    """

    words = [w.lower() for w in sentence.split()]

    any_word = list(word_to_vec_map.keys())[0]
    avg = np.zeros(word_to_vec_map[any_word].shape[0])

    count = 0

    for w in words:
        if w in word_to_vec_map:
            avg += word_to_vec_map[w]
            count += 1

    if count > 0:
        avg = avg / count

    return avg


def model(
    X,
    Y,
    word_to_vec_map,
    learning_rate=0.01,
    num_iterations=400,
):
    """
    Model to train word vector representations in numpy.

    Arguments:
        X -- input data, numpy array of sentences as strings, of shape (m,)
        Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
        word_to_vec_map -- dictionary mapping every word in a vocabulary into its
            50-dimensional vector representation
        learning_rate -- learning_rate for the stochastic gradient descent algorithm
        num_iterations -- number of iterations

    Returns:
        pred -- vector of predictions, numpy-array of shape (m, 1)
        W -- weight matrix of the softmax layer, of shape (n_y, n_h)
        b -- bias of the softmax layer, of shape (n_y,)
    """

    # Get a valid word contained in the word_to_vec_map
    any_word = list(word_to_vec_map.keys())[0]

    # number of training examples
    m = Y.shape[0]
    # number of classes
    n_y = len(np.unique(Y))
    # dimensions of the GloVe vectors
    n_h = word_to_vec_map[any_word].shape[0]

    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))

    # Convert Y to Y_one_hot with n_y classes
    Y_oh = convert_to_one_hot(Y, C=n_y)

    # Optimization loop
    for t in range(num_iterations):
        cost = 0
        dW = 0
        db = 0

        # Loop over the training examples
        for i in range(m):
            # Average the word vectors of the words from the i'th training example
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # Forward propagate the avg through the softmax layer.
            z = W @ avg + b
            a = softmax(z)

            # Add the cost using the i'th training label's one hot representation and
            # "A" (the output of the softmax)
            cost += -np.sum(Y_oh[i] * np.log(a))

            # Compute gradients
            dz = a - Y_oh[i]
            dW += np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
            db += dz

            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db

        assert type(cost) == np.float64, "Incorrect implementation of cost"
        assert cost.shape == (), "Incorrect implementation of cost"

        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map)

    return pred, W, b


def sentences_to_indices(
    X,
    word_to_index,
    max_len,
):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to
    words in the sentences. The output shape should be such that it can be given to
    `Embedding()` (described in Figure 4).

    Arguments:
    X -- array of sentences (strings), of shape (m,)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in
        X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of
        shape (m, max_len)
    """

    # number of training examples
    m = X.shape[0]

    X_indices = np.zeros((m, max_len))

    for i in range(m):
        sentence_words = [w.lower() for w in X[i].split()]

        j = 0
        for w in sentence_words:
            if w in word_to_index:
                X_indices[i, j] = word_to_index[w]
                j += 1

    return X_indices


def pretrained_embedding_layer(
    word_to_vec_map,
    word_to_index,
):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional
    vectors.

    Arguments:
        word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
        word_to_index -- dictionary mapping from words to their indices in
            the vocabulary (400,001 words)

    Returns:
        embedding_layer -- pretrained layer Keras instance
    """
    # adding 1 to fit Keras embedding (requirement)
    vocab_size = len(word_to_index) + 1
    any_word = list(word_to_vec_map.keys())[0]
    # define dimensionality of your GloVe word vectors (= 50)
    emb_dim = word_to_vec_map[any_word].shape[0]

    # Initialize the embedding matrix as a numpy array of zeros.
    emb_matrix = np.zeros((vocab_size, emb_dim))

    # Set each row "idx" of the embedding matrix to be
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct input and output sizes
    # Make it non-trainable.
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=emb_dim,
        trainable=False,
    )
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def Emojify_V2(
    input_shape,
    word_to_vec_map,
    word_to_index,
):
    """
    Function creating the Emojify-v2 model's graph.

    Arguments:
        input_shape -- shape of the input, usually (max_len,)
        word_to_vec_map -- dictionary mapping every word in a vocabulary into its
            50-dimensional vector representation
        word_to_index -- dictionary mapping from words to their indices in the
            vocabulary (400,001 words)

    Returns:
        model -- a model instance in Keras
    """

    ### START CODE HERE ###
    # Define sentence_indices as the input of the graph.
    # It should be of shape input_shape and dtype 'int32'
    # (as it contains indices, which are integers).
    sentence_indices = Input(shape=input_shape, dtype="int32")

    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # Propagate sentence_indices through your embedding layer
    # (See additional hints in the instructions).
    embeddings = embedding_layer(sentence_indices)

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # The returned output should be a batch of sequences.
    X = LSTM(units=128, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(rate=0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # The returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(units=128, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(rate=0.5)(X)
    # Propagate X through a Dense layer with 5 units
    X = Dense(units=5)(X)
    # Add a softmax activation
    X = Activation("softmax")(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)

    return model
