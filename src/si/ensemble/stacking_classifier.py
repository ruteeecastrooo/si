from si.data.dataset import Dataset
import numpy as np
from si.metrics.accuracy import accuracy

class StackingClassifier:

    def __init__(self, models, final_model):
        self.models = models
        self.final_model = final_model

    def fit(self, dataset:Dataset):

        dataset_copy = Dataset(dataset.X, dataset.y, dataset.features, dataset.label)
        for m in self.models:
            m.fit(dataset)
            dataset_copy.X = np.c_[dataset_copy.X, m.predict(dataset)]

        self.final_model.fit(dataset_copy)
        return self



    def predict(self, dataset:Dataset):

        data2 = Dataset(dataset.X, dataset.y, dataset.features, dataset.label)
        for m in self.models:
            data2.X = np.c_[data2.X, m.predict(dataset)]

        return self.final_model.predict(data2)

    def score(self, dataset:Dataset):

        return accuracy(dataset.y, self.predict(dataset))

if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    from si.neighbors.knn_classifier import KNNClassifier
    from si.linear_model.logistic_regression import LogisticRegression

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN and Logistic classifier
    knn = KNNClassifier(k=3)
    lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

    # initialize final model
    final_model = KNNClassifier(k=3)

    # initialize the Stacking classifier
    stacking = StackingClassifier([knn, lg], final_model)

    stacking.fit(dataset_train)

    # compute the score
    score = stacking.score(dataset_test)
    print(f"Score: {score}")

    print(stacking.predict(dataset_test))