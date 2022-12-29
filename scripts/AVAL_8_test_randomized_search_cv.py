# imports
from si.io.csv import read_csv
import numpy as np
from si.linear_model.logistic_regression import LogisticRegression
from si.model_selection.AVAL_8_randomized_search import randomized_search_cv

# datasets
breast_bin_dataset = read_csv('../datasets/breast-bin.csv', features=False, label=True)

# standardization
from sklearn.preprocessing import StandardScaler
breast_bin_dataset.X = StandardScaler().fit_transform(breast_bin_dataset.X)

# parameter distribution
parameter_distribution_ = {
    'l2_penalty': np.linspace(1, 10, 10),
    'alpha':  np.linspace(0.001, 0.0001, 100),
    'max_iter': np.linspace(1000, 2000, 200)
 }

# Logistic regression
lg = LogisticRegression()

# Randomized search
scores = randomized_search_cv(model=lg,
                              dataset=breast_bin_dataset,
                              parameter_distribution= parameter_distribution_,
                              cv= 3,
                              n_iter= 10)

print(scores)

# Mais uma vez, podemos facilmente, a partir da variavel scores, obter
#o " melhor modelo" isto e a melhor combinacao de parameteros (tem o maior valor de score)