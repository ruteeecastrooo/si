import numpy as np

import src.si.io.csv as csv
import pandas as pd

ficheiro = 'C:\\Users\\rutee\\OneDrive\\Ambiente de Trabalho\\sib\\si\\datasets\\iris.csv'
dataset = csv.read_csv(ficheiro, features=True, label=True)

print(dataset.features)
print(dataset.label)
print(dataset.X)
print(dataset.y)

primeira_feature = dataset.X[:, 0]
print(primeira_feature.shape)

ultimas_5_amostras = dataset.X[-5:, :]
media_ultimas_5_amostras = np.nanmean(ultimas_5_amostras, axis=0)
print(media_ultimas_5_amostras)

registos_1_ou_superior = np.all(dataset.X > 1, axis=1)
registos_1_ou_superior = dataset.X[registos_1_ou_superior, :]
print(registos_1_ou_superior.shape)

registos_iris_setosa = dataset.y == 'Iris-setosa'
registos_iris_setosa = dataset.X[registos_iris_setosa, :]
print(registos_iris_setosa)
print(registos_iris_setosa.shape)

