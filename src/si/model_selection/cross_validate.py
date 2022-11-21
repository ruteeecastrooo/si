from typing import Callable, Dict, List

import numpy as np

from si.data.dataset import Dataset
from si.model_selection.split import train_test_split


def cross_validate(model,
                   dataset: Dataset,
                   scoring: Callable = None,
                   cv: int = 3,
                   test_size: float = 0.2) -> Dict[str, List[float]]:
    """
    It performs cross validation on the given model and dataset.
    It returns the scores of the model on the dataset.
    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    scoring: Callable
        The scoring function to use.
    cv: int
        The cross validation folds.
    test_size: float
        The test size.
    Returns
    -------
    scores: Dict[str, List[float]]
        The scores of the model on the dataset.
    """
    scores = {
        'seeds': [],
        'train': [],
        'test': []
    }

    # for each fold
    for i in range(cv):
        # get random seed
        random_state = np.random.randint(0, 1000)

        # store seed
        scores['seeds'].append(random_state)

        # split the dataset
        train, test = train_test_split(dataset=dataset, test_size=test_size, random_state=random_state)

        # fit the model on the train set
        model.fit(train)

        # score the model on the test set
        if scoring is None:

            # store the train score
            scores['train'].append(model.score(train))

            # store the test score
            scores['test'].append(model.score(test))

        else:
            y_train = train.y
            y_test = test.y

            # store the train score
            scores['train'].append(scoring(y_train, model.predict(train)))

            # store the test score
            scores['test'].append(scoring(y_test, model.predict(test)))

    return scores


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.neighbors.knn_classifier import KNNClassifier

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)

    # initialize the KNN
    knn = KNNClassifier(k=3)

    # cross validate the model
    scores_ = cross_validate(knn, dataset_, cv=5)

    # print the scores
    print(scores_)