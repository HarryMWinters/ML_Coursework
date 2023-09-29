from resnets_utils import BatchNormalization
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.initializers import random_uniform
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.models import Model


def identity_block(X, f, filters, initializer=random_uniform):
    """
    Implementation of the identity block as defined in Figure 4

    Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV
            layers of the main path
        initializer -- to set up the initial weights of a layer. Equals to random
            uniform initializer

    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(
        filters=F1,
        kernel_size=1,
        strides=(1, 1),
        padding="valid",
        kernel_initializer=initializer(seed=0),
    )(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Conv2D(
        filters=F2,
        kernel_size=f,
        strides=(1, 1),
        padding="same",
        kernel_initializer=initializer(seed=0),
    )(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Conv2D(
        filters=F3,
        kernel_size=1,
        strides=(1, 1),
        padding="valid",
        kernel_initializer=initializer(seed=0),
    )(X)
    X = BatchNormalization(axis=3)(X)  # Default axis

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


def convolutional_block(
    X,
    f,
    filters,
    s=2,
    initializer=glorot_uniform,
):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV
            layers of the main path
        s -- Integer, specifying the stride to be used
        initializer -- to set up the initial weights of a layer. Equals to Glorot
            uniform initializer, also called Xavier uniform initializer.

    Returns:
        X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
    """

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(
        filters=F1,
        kernel_size=1,
        strides=(s, s),
        padding="valid",
        kernel_initializer=initializer(seed=0),
    )(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Conv2D(
        filters=F2,
        kernel_size=f,
        strides=(1, 1),
        padding="same",
        kernel_initializer=initializer(seed=0),
    )(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Conv2D(
        filters=F3,
        kernel_size=1,
        strides=(1, 1),
        padding="valid",
        kernel_initializer=initializer(seed=0),
    )(X)
    X = BatchNormalization(axis=3)(X)

    X_shortcut = Conv2D(
        filters=F3,
        kernel_size=1,
        strides=(s, s),
        padding="valid",
        kernel_initializer=initializer(seed=0),
    )(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    Stage 1:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL
    Stage 2:
        -> CONVBLOCK -> IDBLOCK*2
    Stage 3:
        -> CONVBLOCK -> IDBLOCK*3
    Stage 4:
        -> CONVBLOCK -> IDBLOCK*5
    Stage 5:
        -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL

    -> FLATTEN -> DENSE

    Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(
        64,
        (7, 7),
        strides=(2, 2),
        kernel_initializer=glorot_uniform(seed=0),
    )(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    ## Stage 3
    X = convolutional_block(
        X,
        f=3,
        filters=[128, 128, 512],
        s=2,
    )
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])

    # Stage 4
    X = convolutional_block(
        X,
        f=3,
        filters=[256, 256, 1024],
        s=2,
    )

    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    # Stage 5
    X = convolutional_block(
        X,
        f=3,
        filters=[512, 512, 2048],
        s=2,
    )

    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])

    X = AveragePooling2D()(X)

    X = Flatten()(X)
    X = Dense(classes, activation="softmax", kernel_initializer=glorot_uniform(seed=0))(
        X
    )

    model = Model(inputs=X_input, outputs=X)

    return model
