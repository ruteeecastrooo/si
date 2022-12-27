#10.1
import numpy as np


class SoftMaxDense:
    """
    A dense layer is a layer where each neuron is connected to all neurons in the previous layer.
    Parameters
    ----------
    input_size: int
        The number of inputs the layer will receive.
    output_size: int
        The number of outputs the layer will produce.
    Attributes
    ----------
    weights: np.ndarray
        The weights of the layer.
    bias: np.ndarray
        The bias of the layer.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the dense layer.
        Parameters
        ----------
        input_size: int
            The number of inputs the layer will receive.
        output_size: int
            The number of outputs the layer will produce.
        """
        # parameters
        self.input_size = input_size
        self.output_size = output_size

        # attributes
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        Parameters
        ----------
        X: np.ndarray
            The input to the layer.
        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """
        Z=np.dot(X, self.weights)
        softmax=SoftMaxActivation.forward(Z + self.bias)
        return softmax

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        """
        return error


#10.1
class SoftMaxActivation:
    """
    A SoftMax Activation layer.
    """

    def __init__(self):
        """
        Initialize the SoftMax Activation layer.
        """
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        Parameters
        ----------
        X: np.ndarray
            The input to the layer.
        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """
        denominator = np.sum(np.exp(X), axis=1, keepdims=True)

        return np.exp(X)/denominator

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        """
        return error