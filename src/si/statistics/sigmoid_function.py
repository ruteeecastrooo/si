import numpy as np


def sigmoid_function(X: np.ndarray) -> np.ndarray:
    """
    It returns the sigmoid function of the given input
    Parameters
    ----------
    X: np.ndarray
        The input of the sigmoid function
    Returns
    -------
    sigmoid: np.ndarray
        The sigmoid function of the given input
    """
    return 1 / (1 + np.exp(-X))