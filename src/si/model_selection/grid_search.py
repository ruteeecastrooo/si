import itertools
from typing import Callable, Tuple, Dict, List, Any

from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate


def grid_search_cv(model,
                   dataset: Dataset,
                   parameter_grid: Dict[str, Tuple],
                   scoring: Callable = None,
                   cv: int = 5,
                   test_size: float = 0.2) -> List[Dict[str, Any]]:
    """
    Performs a grid search cross validation on a model.
    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    parameter_grid: Dict[str, Tuple]
        The parameter grid to use.
    scoring: Callable
        The scoring function to use.
    cv: int
        The cross validation folds.
    test_size: float
        The test size.
    Returns
    -------
    scores: List[Dict[str, List[float]]]
        The scores of the model on the dataset.
    """
    # validate the parameter grid
    for parameter in parameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    scores = []

    # for each combination
    for combination in itertools.product(*parameter_grid.values()):

        # parameter configuration
        parameters = {}

        # set the parameters
        for parameter, value in zip(parameter_grid.keys(), combination):
            setattr(model, parameter, value)
            parameters[parameter] = value

        # cross validate the model
        score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size)

        # add the parameter configuration
        score['parameters'] = parameters

        # add the score
        scores.append(score)

    return scores


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.linear_model.logistic_regression import LogisticRegression

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)

    # initialize the Logistic Regression model
    knn = LogisticRegression()

    # parameter grid
    parameter_grid_ = {
        'l2_penalty': (1, 10),
        'alpha': (0.001, 0.0001),
        'max_iter': (1000, 2000)
    }

    # cross validate the model
    scores_ = grid_search_cv(knn,
                             dataset_,
                             parameter_grid=parameter_grid_,
                             cv=3)

    # print the scores
    print(scores_)