import itertools
import numpy as np
from typing import Callable, Tuple, Dict, List, Any

from si.data.dataset import Dataset
from si.model_selection.cross_validate import cross_validate


def randomized_search_cv(model,
                        dataset: Dataset,
                        #dicionário com nome do parametro e distribuião de valores
                        parameter_distribution: Dict[str, np.ndarray],
                        scoring: Callable = None,
                        cv: int = 5,
                        n_iter: int = 10,
                        test_size: float = 0.2) -> List[Dict[str, Any]]:
    """
    Performs a randomized search cross validation on a model.
    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    parameter_distribution: Dict[str, np.ndarray]
        The parameter distribution to use.
    scoring: Callable
        The scoring function to use.
    cv: int
        The cross validation folds.
    n_iter: int
        the number of random combinations of parameters
    test_size: float
        The test size.
    Returns
    -------
    scores: List[Dict[str, List[float]]]
        The scores of the model on the dataset.
    """
    # validate the parameter distribution
    for parameter in parameter_distribution:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    scores = []

     # parameter configuration
    combination = random_parameter_combination(parameter_distribution)

    # set the parameters

    #for key in parameters:
        #setattr(model, key, parameters[key])
    parameters={}
    for parameter, value in combination:
        print(parameter+":  "+ value)
        setattr(model, parameter, value)
        parameters[parameter] = value

    # cross validate the model
    score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size)

    # add the parameter configuration
    score['parameters'] = parameters

    # add the score
    scores.append(score)

    return scores



#auxiliar functions

#ex8
def random_element(values):
    random_index = np.random.randint(0, high=len(values))
    return values[random_index]


def random_parameter_combination(parameter_distribution):
    results = {}

    for key in parameter_distribution:
        values = parameter_distribution[key]
        results[key] = random_element(values)

    return results


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.linear_model.logistic_regression import LogisticRegression

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)

    # initialize the Logistic Regression model
    lg = LogisticRegression()

    # parameter distribution
    parameter_distribution_ = {
        'l2_penalty': np.linspace(1, 10, 10),
        'alpha':  np.linspace(0.001, 0.0001, 100),
        'max_iter': np.linspace(1000, 2000, 200),

    }

    # cross validate the model
    scores_ = randomized_search_cv(lg, dataset_, parameter_distribution=parameter_distribution_)

    # print the scores
    print(scores_)
