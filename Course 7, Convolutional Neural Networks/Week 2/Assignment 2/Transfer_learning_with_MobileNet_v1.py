import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation


def data_augmenter():
    """
    Create a Sequential model composed of 2 layers
    Returns:
        tf.keras.Sequential
    """
    data_augmentation = tf.keras.Sequential(
        [
            RandomFlip("horizontal"),
            RandomRotation(0.2),
        ]
    )

    return data_augmentation


IMG_SIZE = (160, 160)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


def alpaca_model(
    image_shape=IMG_SIZE,
    data_augmentation=data_augmenter(),
):
    """Define a tf.keras model for binary classification out of the MobileNetV2 model
    Arguments:
        image_shape -- Image width and height
        data_augmentation -- data augmentation function
    Returns:
    Returns:
        tf.keras.model
    """

    input_shape = image_shape + (3,)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)

    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tfl.GlobalAveragePooling2D()(x)
    x = tfl.Dropout(0.2)(x)

    outputs = tfl.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)

    return model


def transfer_model(donor_model, base_learning_rate):
    base_model = donor_model.layers[4]
    base_model.trainable = True

    # Fine-tune from this layer onwards
    fine_tune_at = 120

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1 * base_learning_rate)
    metrics = ["accuracy"]

    donor_model.compile(
        loss=loss_function,
        optimizer=optimizer,
        metrics=metrics,
    )

    return donor_model
