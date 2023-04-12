import tensorflow as tf

num_actions = 4
state_size = 8
ALPHA = 0.1

# Create the Q-Network
q_network = tf.keras.Sequential(
    [
        tf.keras.layers.Input(input_shape=state_size),
        tf.keras.layers.Dense(units=64, activation="relu"),
        tf.keras.layers.Dense(units=64, activation="relu"),
        tf.keras.layers.Dense(units=num_actions, activation="linear"),
    ]
)

# Create the target Q^-Network
target_q_network = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=64, activation="relu"),
        tf.keras.layers.Dense(units=64, activation="relu"),
        tf.keras.layers.Dense(units=num_actions, activation="linear"),
    ]
)

optimizer = tf.optimizers.Adam(learning_rate=ALPHA)


# ------------------------------------------------------------
from collections import namedtuple

experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "next_state", "done"]
)


def compute_loss(
    experiences: experience,
    gamma: float,
    q_network: tf.keras.Sequential,
    target_q_network: tf.keras.Sequential,
) -> tf.Tensor:
    """
    Calculates the loss.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values
      target_q_network: (tf.keras.Sequential) Keras model for predicting the targets

    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """

    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences

    # Compute max Q^(s,a)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)

    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = tf.where(done_vals == 1, rewards, rewards + gamma * max_qsa)

    # Get the q_values and reshape to match y_targets
    q_values = q_network(states)
    q_values = tf.gather_nd(
        q_values,
        tf.stack(
            [
                tf.range(q_values.shape[0]),
                tf.cast(actions, tf.int32),
            ],
            axis=1,
        ),
    )

    # Compute the loss
    MSE = tf.keras.losses.MSE
    loss = MSE(y_targets, q_values)

    return loss
