from collections.abc import Callable
import numpy as np
import pandas as pd
from sklearn import preprocessing

def randomized_search_cv(
                   model,
                   dataset,
                   scoring: Callable = None,
                   parameter_distribution = dict,
                   cv: int = 3,
                   n_iter: int = 10,
                   test_size: float = 0.2) -> dict[str, list[float]]:
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
    # Verifica se os parâmetros fornecidos existem no modelo
    for parameter in parameter_distribution:
            if not hasattr(model, parameter):
                raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    scores = []
    # Obtém n_iter combinações de parâmetros

    for _ in range(n_iter):
        parameters = {k: np.random.choice(v) for k,v in parameter_distribution.items()}

        for param, val in parameters.items():
            print(param, val)
            setattr(model, param, val)
        print(dataset.y)
        score = cross_validate(model, dataset, scoring, cv, test_size)
        score["parameters"] = parameters


        scores.append(score)

    return scores

if __name__ == '__main__':
    from si.data.dataset import Dataset
    from si.model_selection.cross_validate import cross_validate
    from si.linear_model.logistic_regression import LogisticRegression

    # load and split the dataset
    df = pd.read_csv("C:\\Users\\rutee\\OneDrive\\Ambiente de Trabalho\\sib\\si\\datasets\\breast-bin.data")
    breast_dataset = Dataset.from_dataframe(df)


    breast_dataset.X = preprocessing.StandardScaler().fit_transform(breast_dataset.X)
    # initialize the Logistic Regression model
    knn = LogisticRegression()

    # parameter grid
    parameter_grid_ = {
        'l2_penalty': (1, 10),
        'alpha': (0.001, 0.0001),
        'max_iter': (1000, 2000)
    }

    # cross validate the model
    scores_ = randomized_search_cv(knn,
                             breast_dataset,
                             parameter_distribution=parameter_grid_,
                             cv=3)

    # print the scores
    print(scores_)
