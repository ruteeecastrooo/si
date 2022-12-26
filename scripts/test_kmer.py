# imports
from si.io.csv import read_csv
from si.linear_model.logistic_regression import LogisticRegression
from si.feature_extraction.k_mer import KMer

# 1. datasets
dataset = read_csv('../datasets/tfbs.csv', features=False, label=True)
#2
k_mer_ = KMer(k=3)
dataset_ = k_mer_.fit_transform(dataset)
print(dataset_.X)
print(dataset_.features)
# 3.standardization
from sklearn.preprocessing import StandardScaler
dataset_.X = StandardScaler().fit_transform(dataset_.X)


from si.model_selection.split import train_test_split

# 4.load and split the dataset
dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

#5.
lg=LogisticRegression()
lg.fit(dataset_train)

# 6. compute the score
score = lg.score(dataset_test)
print(f"Score: {score}")
