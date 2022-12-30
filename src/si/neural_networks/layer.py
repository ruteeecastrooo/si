import numpy as np


class Dense:
    """
    A dense layer is a layer where each neuron is connected to all neurons in the previous layer.
    Parameters
    ----------
    input_size: int
        The number of inputs the layer will receive.
    output_size: int
        The number of outputs the layer will produce.
    Attributes
    ----------
    weights: np.ndarray
        The weights of the layer.
    bias: np.ndarray
        The bias of the layer.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the dense layer.
        Parameters
        ----------
        input_size: int
            The number of inputs the layer will receive.
        output_size: int
            The number of outputs the layer will produce.
        """
        # parameters
        self.input_size = input_size
        self.output_size = output_size

        # attributes
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

        # podemos criar mais uma variável na qual guardamos
        #os últimos X's recebidos no forward
        self.X = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        Parameters
        ----------
        X: np.ndarray
            The input to the layer.
        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """
        self.X = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:


        # 1) quais são as dimensões da matriz error?
        # - numero de colunas = output_size

        # 2) Actualizar parametros da rede: WEIGHTS e BIAS
        # - actualizar os parametros da rede associados a este layers
        # -- self.weights
        # -- self.bias
        #
        # Relembrar fórmula geral da actualização:
        #   W = W - learning_rate*derivativeon_W
        #   B = B - learning_rate*derivativeon_B
        # 3) Usando a teoria, a regra da cadeia e o erro acumulado
        # que recebemos, vindo do layers imediatamente à direita de nós
        # podemos facilmente actualizar estes parametros.
        #  - recisamos apenas de acessos aos X's que recebemos
        #como input no metodo forward


        # ---------------------------------------------------------------------------
        #
        #    -> actualziar os bias (deste layer, l)
        #    -> actualziar os pesos
        #    -> acumula erro e devovle essa acumulacao

        # 1) Actualizar todos os Bias dos nós (deste layer)
        derivative_of_error_in_order_of_bias = error
        self.bias = self.bias - derivative_of_error_in_order_of_bias

        # 2) Actualizar todos os pesos de arestas, que partem do layer a quem vamos
        # o erro acumulado e terminam em nós deste layer

        derivative_of_error_in_order_of_weights = np.dot(np.transpose(self.X), error)
        self.weights = self.weights - derivative_of_error_in_order_of_weights

        # 3) Actualizar o Erro acumulado, isto é, criar E^{l} a partir do input
        # que recebemos error=E^{l+1} e passamos esta informacao ao layer l-1
        # para ele fazer o seu trabalho
        derivative_of_error_in_order_of_X = np.dot(error, np.transpose(self.weights))
        new_error = derivative_of_error_in_order_of_X

        return new_error