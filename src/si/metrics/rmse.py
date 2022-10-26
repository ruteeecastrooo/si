import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.sum(np.power(y_true-y_pred, 2)) / len(y_true))
