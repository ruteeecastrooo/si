import numpy as np

class Dataset():

    # X: matriz dos dados propriamente ditos                                                                   (100,3)
    # y: vetor (ou lista) de tamanho "nº de exemplos" que indica a que classe (label) pertence aquele exemplo  (100)
    # features: vetor (ou lista) com o nome de cada coluna (feature)                                           (3)
    # label: string nome do vetor da variável dependente (do label/variável de interesse)
    def __init__(self, X: np.array, y: list, features: list[str] = None, label: str = None) -> None:
        self.X = X
        self.y = y
        self.features = features
        self.label = label
    
    def shape(self):
        return self.X.shape

    def has_label(self) -> bool:
        return self.label != None
        # if self.label != None:
        #     return True
        # else:
        #     return False

    def get_classes(self):
        return list(set(self.y))

    # TODO
    # - get_mean, get_variance, get_median, get_min, get_max – devolve média,
    # variância, mediana, valor mínimo e máximo para cada feature/variável
    # dependente
    # - summary – devolve um pandas DataFrame com todas as métricas descritivas


if __name__== "__main__":
    X=np.array([[1,2,3],[1,1,1],[5,1,7],[9,1,5]])
    y=np.array([0,0,1,1])
    features=["A","B","C"]
    label="D"
    d=Dataset(X=X,y=y, features=features, label=label)
    print(d.shape())
    print(d.has_label())
    print(d.get_classes())