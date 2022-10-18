import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    It computes the euclidean distance of a point (x) to a set of points y.
    distance_y1n = sqrt((x1 - y11)^2 + (x2 - y12)^2 + ... + (xn - y1n)^2)
    distance_y2n = sqrt((x1 - y21)^2 + (x2 - y22)^2 + ... + (xn - y2n)^2)
    ...
    Parameters
    ----------
    x: np.ndarray
        Point.
    y: np.ndarray
        Set of points.
    Returns
    -------
    np.ndarray
        Euclidean distance for each point in y.
    """
    return np.sqrt(((x - y) ** 2).sum(axis=1))
