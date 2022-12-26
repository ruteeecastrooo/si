# imports
from si.io.csv import read_csv
import numpy as np
from si.linear_model.logistic_regression import LogisticRegression
from si.model_selection.cross_validate import cross_validate
from si.model_selection.randomized_search import randomized_search_cv

# datasets
breast_bin_dataset = read_csv('../datasets/iris.csv', features=False, label=True)

# standardization
from sklearn.preprocessing import StandardScaler
breast_bin_dataset.X = StandardScaler().fit_transform(breast_bin_dataset.X)

# randomized search cv
lg = LogisticRegression()

# parameter distribution
parameter_distribution_ = {
    'l2_penalty': np.linspace(1, 10, 10),
    'alpha':  np.linspace(0.001, 0.0001, 100),
    'max_iter': np.linspace(1000, 2000, 200)
 }


# cross validate the model
scores = randomized_search_cv(model=lg,
                              dataset=breast_bin_dataset,
                              parameter_distribution= parameter_distribution_,
                              cv= 3,
                              n_iter= 10)

#print(scores)

#show scores as dataframe

#import pandas as pd
#cols = list(scores[0]['parameters'].keys())
#cols = cols + ['train', 'test', 'cv']

#dict_df = {col: [] for col in cols}
#for score in scores:
    #for i, (train_val, test_val) in enumerate(zip(score['train'], score['test'])):
        #dict_df['cv'].append(i)
        #dict_df['train'].append(train_val)
        #dict_df['test'].append(test_val)
        #for p_key, p_val in score['parameters'].items():
            #dict_df[p_key].append(p_val)

#df = pd.DataFrame(dict_df)
#print(df)