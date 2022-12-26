import numpy as np
class RoundActivation:
    """
    A Round Activation layer.
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

        return X.round()

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        """
        return error