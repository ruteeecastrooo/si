from typing import Callable

import numpy as np
import pandas as pd

from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile:
    def __init__(self, score_func: Callable = f_classification, percentile: float = 0.5): #queremos manter metade
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectPercentile':

        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        # self.F = [10, 5, 1]
        # [1,5,10]
        # [2,1,0]
        num_total = len(list(dataset.features))
        num_a_manter = int(num_total * self.percentile)
        idxs = np.argsort(self.F)[-num_a_manter:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    from si.data.dataset import Dataset

    df = pd.read_csv("C:\\Users\\rutee\\OneDrive\\Ambiente de Trabalho\\sib\\si\\datasets\\iris.csv")
    dataset = Dataset.from_dataframe(df, label='class')
    # dataset = Dataset(X=np.array([[0, 2, 0, 3],
    #                               [0, 1, 4, 3],
    #                               [0, 1, 1, 3]]),
    #                   y=np.array([0, 1, 0]),
    #                   features=["f1", "f2", "f3", "f4"],
    #                   label="y")

    selector = SelectPercentile(percentile=0.5)
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)
