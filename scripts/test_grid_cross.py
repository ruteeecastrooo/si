# imports
from si.io.csv import read_csv
from si.linear_model.logistic_regression import LogisticRegression
from si.model_selection.cross_validate import cross_validate
from si.model_selection.grid_search import grid_search_cv

# datasets
breast_bin_dataset = read_csv('../datasets/breast-bin.csv', features=False, label=True)
# standardization
from sklearn.preprocessing import StandardScaler
breast_bin_dataset.X = StandardScaler().fit_transform(breast_bin_dataset.X)
# cross validation
lg = LogisticRegression()
print("cross validate")
scores = cross_validate(lg, breast_bin_dataset, cv=5)
print(scores)

lg = LogisticRegression()

# parameter grid
parameter_grid = {
    'l2_penalty': (1, 10),
    'alpha': (0.001, 0.0001, 0.00001),
    'max_iter': (1000, 2000, 3000, 4000, 5000, 6000)
}

# cross validate the model
scores = grid_search_cv(lg,
                        breast_bin_dataset,
                        parameter_grid=parameter_grid,
                        cv=3)
print("grid_search")
print(scores)
