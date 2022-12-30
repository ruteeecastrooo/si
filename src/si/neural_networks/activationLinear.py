
import numpy as np

#10.5
class LinearActivation:
    """
    A linear activation layer.
    """

    def __init__(self):
        """
        Initialize the ReLU activation layer.
        """
        self.X = []

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
        self.X = X
        return X

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:

        return error
