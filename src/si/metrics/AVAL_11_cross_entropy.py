import numpy as np

def cross_entropy (y_true: np.ndarray, y_pred: np.ndarray) -> float:

    return -np.sum(y_true * np.log(y_pred)) / len(y_true)

def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:

    # array with len(y) lines and 1 column
    ones=np.ones((len(y_true),1))
    return np.divide(-ones,y_pred*len(y_true))