from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.metrics.mse import mse, mse_derivative


class NN:
    """
    The NN is the Neural Network model.
    It comprehends the model topology including several neural network layers.
    The algorithm for fitting the model is based on backpropagation.
    Parameters
    ----------
    layers: list
        List of layers in the neural network.
    epochs: int
        Number of epochs to train the model.
    learning_rate: float
        The learning rate of the model.
    loss: Callable
        The loss function to use.
    loss_derivative: Callable
        The derivative of the loss function to use.
    verbose: bool
        Whether to print the loss at each epoch.
    Attributes
    ----------
    history: dict
        The history of the model training.
    """
    def __init__(self,
                 layers: list,
                 epochs: int = 1000,
                 learning_rate: float = 0.01,
                 loss: Callable = mse,
                 loss_derivative: Callable = mse_derivative,
                 verbose: bool = False):
        """
        Initialize the neural network model.
        Parameters
        ----------
        layers: list
            List of layers in the neural network.
        epochs: int
            Number of epochs to train the model.
        learning_rate: float
            The learning rate of the model.
        loss: Callable
            The loss function to use.
        loss_derivative: Callable
            The derivative of the loss function to use.
        verbose: bool
            Whether to print the loss at each epoch.
        """
        # parameters
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.loss_derivative = loss_derivative
        self.verbose = verbose

        # attributes
        self.history = {}

    def fit(self, dataset: Dataset) -> 'NN':
        """
        It fits the model to the given dataset.
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        Returns
        -------
        self: NN
            The fitted model
        """
        X = dataset.X
        y = dataset.y

        for epoch in range(1, self.epochs + 1):
            #print("=============================")
            #print("\t\t Epoch = " + str(epoch))
            #print("=============================")

            X_layers = X
            # forward propagation
            #print("=============================")
            #print("\t\t Forward")
            for layer in self.layers:

                #print("=============================")
                #print("testar input dos layers")

                X_layers = layer.forward(X_layers)

            #print()
            #print("Y_pred:")
            #print(X_layers)
            #print()

            # in the end X_layers is y_pred

            # backward propagation
            #print("=============================")
            #print("\t\t Backpropagation")
            error = self.loss_derivative(y, X_layers)
            #print("=============================")
            #print("Loss Derivative Error ")
            #print(error)
            #print("=============================")
            for layer in self.layers[::-1]:
                error = layer.backward(error, self.learning_rate)
                #print("updated error")
                #print(error)

            # save history
            cost = self.loss(y, X_layers)
            #print("=============================")
            #print("\t\t Cost")
            self.history[epoch] = cost

            # print loss
            if self.verbose:
                print(f'Epoch {epoch}/{self.epochs} - cost: {cost}')

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the output of the given dataset.
        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of
        Returns
        -------
        predictions: np.ndarray
            The predicted output
        """
        X = dataset.X

        # forward propagation
        for layer in self.layers:
            X = layer.forward(X)

        return X

    def cost(self, dataset: Dataset) -> float:
        """
        It computes the cost of the model on the given dataset.
        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost on
        Returns
        -------
        cost: float
            The cost of the model
        """
        y_pred = self.predict(dataset)
        return self.loss(dataset.y, y_pred)

    def score(self, dataset: Dataset, scoring_func: Callable = accuracy) -> float:
        """
        It computes the score of the model on the given dataset.
        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the score on
        scoring_func: Callable
            The scoring function to use
        Returns
        -------
        score: float
            The score of the model
        """
        y_pred = self.predict(dataset)
        return scoring_func(dataset.y, y_pred)


if __name__ == '__main__':
    pass

#The ideia to solve the missing 2 exercises (12.2 and 12.3) is to study the
#dataset, that is:

#i) Find out the number of features it has and create that many nodes for the input layer.

   #- That is, if |features| = 10, then the first dense layer would be l1= Dense(10, _ )

#ii) Find wich problem we are dealing with, namely if it is a regression or a classification problem.

#ii.1)In case it is a regression problem we could create an output dense layer dense as we did in exercise 10.5, and it's actiation layer could be LinearActivation (also, as in 10.5) or a function that would yield values more suited to the desired range of values.

#ii.2) Otherwise, for a classification problem we can split it in 2 types:
#   ii.2.a) Binary classification: where could have an output layer (dense + activation) just like we did in 10.3
#   ii.2.b) Multiclass classification: where could use 1-HOT encoding, that is, have an output layer (dense + activation) just like we did in 10.4.

#Regarding the other parameters of the neural network:

#- We could have any number of layers: the best option could only be decided after
#by comparison using some pre-determined metrics (ex: accucary).

#- The same ideia could be applied to the loss function although, in this case,
#some functions are more well suited for the type of problem we are dealing with.

#- There's also no deterministic rule to find "`a priori" the best number of
#epochs. One must take into consideration to properly train the network but
#without making it a "training_dataset"-only expert, falling into the trap
#of overfitting.

#* In conclusion, after inspecting the datasets, one needs only to:

#1) read the dataset
#2) do some preprocessing (if needed)
#3) split the dataset into: training_dataset and testing_dataset
#4) create a suited neural network for it's the problem (already explained above):
#   4.1) train the neural network (using the training_dataset)
#   4.2) do predictions on the testing_dataset (using the already trained neural network)
#5) Calcule some metrics regarding the predictions in 4.2)
#6) If not satisfied, either re-train the network or create a different one and go back to step 4)