import numpy as np
class SigmoidActivation:
    """
    A sigmoid activation layer.
    """

    def __init__(self):
        """
        Initialize the sigmoid activation layer.
        """
        # podemos criar mais uma variável na qual guardamos
        # os últimos X's recebidos no forward
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
        return 1 / (1 + np.exp(-X))


    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        derivative_of_sigmoid = self.forward(self.X) * (1 - self.forward(self.X))

        return error * derivative_of_sigmoid