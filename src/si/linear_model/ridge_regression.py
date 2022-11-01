import numpy as np

from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegression:
    """
    The RidgeRegression is a linear model using the L2 regularization.
    This model solves the linear regression problem using an adapted Gradient Descent technique

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    alpha: float
        The learning rate
    max_iter: int
        The maximum number of iterations

    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the linear model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
        For example, theta_zero * 1
    """
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000, use_adaptive_alpha: bool = False):
        """

        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        alpha: float
            The learning rate
        max_iter: int
            The maximum number of iterations
        """
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        # attributes
        self.theta = None
        self.theta_zero = None

        self.cost_history = {}
        self.tol_stop = 0.1
        self.tol_adjust_learning_rate = 1
        self.use_adaptive_alpha = use_adaptive_alpha

    def fit(self, dataset: Dataset) -> 'RidgeRegression':
        """
        Fit the model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RidgeRegression
            The fitted model
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # gradient descent
        for i in range(self.max_iter):
            # predicted y
            y_pred = np.dot(dataset.X, self.theta) + self.theta_zero

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            self.cost_history[i] = self.cost(dataset)

            if i != 0:
                if self.cost_history[i-1] - self.cost_history[i] < self.tol_stop:
                    break

            if i != 0 and self.use_adaptive_alpha:
                if self.cost_history[i - 1] - self.cost_history[i] < self.tol_adjust_learning_rate:
                    self.alpha = self.alpha / 2

        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """
        return np.dot(dataset.X, self.theta) + self.theta_zero

    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on

        Returns
        -------
        mse: float
            The Mean Square Error of the model
        """
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost function on

        Returns
        -------
        cost: float
            The cost function of the model
        """
        y_pred = self.predict(dataset)
        return (np.sum((y_pred - dataset.y) ** 2) + (self.l2_penalty * np.sum(self.theta ** 2))) / (2 * len(dataset.y))


if __name__ == '__main__':
    # # import dataset
    # from si.data.dataset import Dataset
    #
    # # make a linear dataset
    # X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = np.dot(X, np.array([1, 2])) + 3
    # dataset_ = Dataset(X=X, y=y)
    #
    # # fit the model
    # model = RidgeRegression()
    # model.fit(dataset_)
    #
    # # get coefs
    # print(f"Parameters: {model.theta}")
    #
    # # compute the score
    # score = model.score(dataset_)
    # print(f"Score: {score}")
    #
    # # compute the cost
    # cost = model.cost(dataset_)
    # print(f"Cost: {cost}")
    #
    # # predict
    # y_pred_ = model.predict(Dataset(X=np.array([[3, 5]])))
    # print(f"Predictions: {y_pred_}")

    ####################
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    df = pd.read_csv("C:\\Users\\rutee\\OneDrive\\Ambiente de Trabalho\\sib\\si\\datasets\\cpu.csv")
    print(df.head())
    dataset_ = Dataset.from_dataframe(df, label='perf')
    dataset_.X = StandardScaler().fit_transform(dataset_.X)

    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.25)

    # fit the model
    model = RidgeRegression(max_iter=3000)
    model.fit(dataset_train)

    # get coefs
    print(f"Parameters: {model.theta}")

    # compute the score
    score_train = model.score(dataset_train)
    print(f"Score train: {score_train}")
    score_test = model.score(dataset_test)
    print(f"Score test: {score_test}")

    # compute the cost
    cost_train = model.cost(dataset_train)
    print(f"Cost train: {cost_train}")
    cost_test = model.cost(dataset_test)
    print(f"Cost test: {cost_test}")

    # predict
    y_pred_ = model.predict(dataset_test)
    print(f"Predictions: {y_pred_}")

    import matplotlib
    matplotlib.use('TkAgg')

    plt.plot(model.cost_history.values())
    plt.ylabel("Custo")
    plt.xlabel("Iteração")
    plt.show()

