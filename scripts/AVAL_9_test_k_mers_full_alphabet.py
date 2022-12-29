# imports
from si.io.csv import read_csv
from si.linear_model.logistic_regression import LogisticRegression
from si.feature_extraction.AVAL_9_k_mer import KMer9
from sklearn.preprocessing import StandardScaler
from si.model_selection.split import train_test_split

# 1. load dataset
dataset = read_csv('../datasets/transporters.csv', features=False, label=True)
#inspecionamos o dataset e só tem letras do alfabeto ingles
english_alphabet = "AEIOUBCDFGHJLMNPQRSTVXZKYW"

#2 kmers k=2 en alphabet
k_mer_ = KMer9(k=2, alphabet=english_alphabet)
dataset_ = k_mer_.fit_transform(dataset)
print(dataset_.X)
print(dataset_.features)

# 3.standardization
dataset_.X = StandardScaler().fit_transform(dataset_.X)

# 4.split the dataset
dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

#5. TRAIN lg
lg=LogisticRegression()
lg.fit(dataset_train)

# 6. compute the score
score = lg.score(dataset_test)
print(f"Score: {score}")
