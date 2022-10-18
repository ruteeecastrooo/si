from typing import Callable
import numpy as np
from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance
from numpy.linalg import svd


class PCA:
    def __init__(self, n_components: int):
        # qts colunas vai ter o output pedimos 3
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    # estima a média, os componentes e a variância explicada
    def fit(self, dataset: Dataset) -> 'PCA':
        self.mean = np.mean(dataset.X, axis=0)
        X_tmp = dataset.X - self.mean

        U, S, V_T = svd(X_tmp, full_matrices=False)

        self.components = V_T[0:self.n_components]
        num_amostras = dataset.X.shape[0]
        tmp_explained_variance = S**2 / (num_amostras-1)

        self.explained_variance = sum(tmp_explained_variance[0:self.n_components]) / sum(tmp_explained_variance)

        return self

    def transform(self, dataset: Dataset) -> np.ndarray:
        X_tmp = dataset.X - self.mean

        X_reduced = np.dot(X_tmp, np.transpose(self.components))
        return X_reduced

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """
        It fits and transforms the dataset.
        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        Returns
        -------
        np.ndarray
            Transformed dataset.
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    from si.data.dataset import Dataset
    import pandas as pd
    from sklearn import preprocessing
    # dataset_ = Dataset.from_random(100, 5)
    df = pd.read_csv("C:\\Users\\rutee\\OneDrive\\Ambiente de Trabalho\\sib\\si\\datasets\\iris.csv")
    print(df.head())
    dataset_ = Dataset.from_dataframe(df, label='class')
    dataset_.X = preprocessing.scale(dataset_.X)

    n = 2
    pca = PCA(n)
    res = pca.fit_transform(dataset_)
    print(res.shape)

    print(pca.explained_variance)

