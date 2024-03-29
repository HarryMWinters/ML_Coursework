import numpy as np
from gc_utils import dictionary_to_vector
from gc_utils import gradients_to_vector
from gc_utils import relu
from gc_utils import sigmoid
from gc_utils import vector_to_dictionary
from public_tests import *
from testCases import *

# GRADED FUNCTION: forward_propagation


def forward_propagation(x, theta):
    """
    Implement the linear forward propagation (compute J) presented in Figure 1 (J(theta) = theta * x)

    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well

    Returns:
    J -- the value of function J, computed using the formula J(theta) = theta * x
    """

    J = theta * x

    return J


def backward_propagation(x, theta):
    """
    Computes the derivative of J with respect to theta (see Figure 1).

    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well

    Returns:
    dtheta -- the gradient of the cost with respect to theta
    """

    dtheta = x

    return dtheta


def gradient_check(
    x,
    theta,
    epsilon=1e-7,
    print_msg=False,
):
    """
    Implement the gradient checking presented in Figure 1.

    Arguments:
    x -- a float input
    theta -- our parameter, a float as well
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient. Float output
    """

    theta_plus = theta + epsilon
    theta_minus = theta - epsilon
    J_plus = forward_propagation(x, theta_plus)
    J_minus = forward_propagation(x, theta_minus)
    gradapprox = (J_plus - J_minus) / (2 * epsilon)

    grad = backward_propagation(x, theta)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if print_msg:
        if difference > 2e-7:
            print(
                "\033[93m"
                + "There is a mistake in the backward propagation! difference = "
                + str(difference)
                + "\033[0m"
            )
        else:
            print(
                "\033[92m"
                + "Your backward propagation works perfectly fine! difference = "
                + str(difference)
                + "\033[0m"
            )

    return difference


def forward_propagation_n(X, Y, parameters):
    """
    Implements the forward propagation (and computes the cost) presented in Figure 3.

    Arguments:
    X -- training set for m examples
    Y -- labels for m examples
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (5, 4)
                    b1 -- bias vector of shape (5, 1)
                    W2 -- weight matrix of shape (3, 5)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)

    Returns:
    cost -- the cost function (logistic cost for m examples)
    cache -- a tuple with the intermediate values (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    """

    # retrieve parameters
    m = X.shape[1]

    W1 = parameters["W1"]
    b1 = parameters["b1"]

    W2 = parameters["W2"]
    b2 = parameters["b2"]

    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # Cost
    log_probs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)

    cost = 1.0 / m * np.sum(log_probs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache


def gradient_check_n(
    parameters,
    gradients,
    X,
    Y,
    epsilon=1e-7,
    print_msg=False,
):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters
    X -- input datapoint, of shape (input size, number of examples)
    Y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """

    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)

    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # Compute gradapprox
    for i in range(num_parameters):
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have outputs two parameters but we only
        # care about the first one

        theta_plus = np.copy(parameters_values)
        theta_plus[i] = theta_plus[i] + epsilon
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_plus))

        theta_minus = np.copy(parameters_values)
        theta_minus[i] = theta_minus[i] - epsilon
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_minus))

        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if print_msg:
        if difference > 2e-7:
            print(
                "\033[93m"
                + "There is a mistake in the backward propagation! difference = "
                + str(difference)
                + "\033[0m"
            )
        else:
            print(
                "\033[92m"
                + "Your backward propagation works perfectly fine! difference = "
                + str(difference)
                + "\033[0m"
            )

    return difference
