
import numpy as np

# AVAL 10.2
class ReLUActivation:
    """
    A ReLU activation layer.
    """

    def __init__(self):
        """
        Initialize the ReLU activation layer.
        """
        self.X = []

    # AVAL 10.2
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
        return np.maximum(0,X)

    # AVAL 12.1
    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:

        #for row in error:
        #    for i in range(len(row)):
        #        if row[i] > 0:
        #            row[i] = 1
        #        else:
        #            row[i] = 0
        error_altered = np.where(error > 0, error, 0)
        return error_altered * self.X