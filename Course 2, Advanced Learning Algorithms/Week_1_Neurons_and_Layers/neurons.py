"""
This is just a mess around file to see how hard coding a neuron would be. 
Obviously it's missing back-prop which is the trickiest part :)
"""
import typing as t

import numpy as np

C = t.TypeVar("C")
R = t.TypeVar("R")


class Neuron:
    """
    A neuron with an input, an output and a way to feed forward.
    """

    def __init__(
        self,
        weights: np.ndarray[t.Tuple[int], np.dtype[np.float64]],
        bias: float,
    ) -> None:
        self.weights = weights
        self.bias = bias

    @staticmethod
    def _sigmoid(x: np.ndarray[t.Tuple[int], np.dtype[np.float64]]) -> float:
        """
        Sigmoid activation function: f(x) = 1 / (1 + e^(-x)).
        """
        return float(1 / (1 + np.exp(-x)))

    def feed_forward(
        self,
        inputs: np.ndarray[t.Tuple[int], np.dtype[np.float64]],
    ) -> float:
        """Calculate activation output given the inputs."""
        total = np.dot(self.weights, inputs) + self.bias
        return self._sigmoid(total)

    def feed_back(self) -> None:
        """
        Update the weights and bias wrt the error.

        TODO Not implemented yet.
        """
        raise NotImplementedError
