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
        #os últimos F's com que trabalhámos
        self.Fs = []

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
        self.Fs = X
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
        #   w=w+ learning_rate*derivativeon_w
        #   b=b+ learning_rate*derivativeon_b
        # 3) Usando a teoria, a regra da cadeia e o erro acumulado
        # que recebemos, vindo do layers imediatamente à direita de nós
        # podemos facilmente actualizar estes parametros.
        #  - recisamos apenas de acessos aos F's


        # E - error da derivada
        #       -> tamanho = numero de nós no layer output
        #
        # Layer output recebe no seu backtrack(E) e vai ter que fazer contas
        #       -> actualiza os seus bias
        #           -> actualiza os seus pesos
        #           -> acumula erro e devovle essa acumulacao
        #
        # Array de erro que o layer l+1 vai proporcionar ao layer l
        #   RETURN E^{l+1} = [[Soma(E) * z_1, Soma(E) * z_2, ..., Soma(E) * z_m]]
        #       -> tamanho = numero de nós dele (ou seja, numero de nos do layer l+1)
        #
        #
        # ---------------------------------------------------------------------------
        # Agora estamos no layer l, ele recebe no seu backtrack(E^{l+1})
        # Portanto cá dentro ele tem acesso ao E^{l+1} = [[Soma(E) * z_1, Soma(E) * z_2, ..., Soma(E) * z_m]]
        #
        #    -> actualziar os bias (deste layer, l)
        #    -> actualziar os pesos
        #    -> acumula erro e devovle essa acumulacao
        #
        # Array de erro que o layer l vai proporcionar ao layer seguinte, l-1
        #   RETURN E^{l} = [[Soma(E^{l+1}) * z_1, Soma(E^{l+1}) * z_2, ..., Soma(E^{l+1}) * z_n]]
        #       -> tamanho = numero de nós dele (ou seja, numero de nos do layer l+1)
        #



        # Some dos erros,será utilizada nos passos abaixo:
        #soma_dos_erros = np.sum(error)

        # 1) Actualizar todos os Bias dos nós (deste layer)
        #for i in range(len(self.bias[0])):
        #    self.bias[0][i] = self.bias[0][i] - (soma_dos_erros) * 1


        # 2) Actualizar todos os pesos de arestas, que partem do layer a quem vamos
        # o erro acumulado e terminam em nós deste layer

        #for r in range(len(self.weights)):  # andar nas linhas da matrix
        #    for c in range(len(self.weights[0])):  # andar nas colunas
        #        self.weights[r][c] = self.weights[r][c] - (soma_dos_erros) * self.Fs[0][r]

        # 3) Actualizar o Erro acumulado, isto é, criar E^{l} a partir do input
        # que recebemos error=E^{l+1} e passamos esta informacao ao layer l-1
        # para ele fazer o seu trabalho
        novo_E = np.zeros((1, self.output_size)) # numero de nos da esquerda
        self.actualiza_novo_E(error, novo_E)
        return novo_E


    # Esta funcao recebe o array error que guarda todos
    # os valores de erros acumulados para os nos da direita.
    # Recebe, de seguida, o array -  novo_E - cujo valor vai ser actualizado,
    # passando a guardar o valor acumulado para o no da esquerda especifica pelo ultimo agumento desta funcao, chamado esq.

    def actualiza_novo_E_na_posicao(self, error: np.ndarray, novo_E: np.ndarray, esq:int):
        for dir in range(len(error)):
            novo_E[esq][0] = novo_E[esq][0] + (error[dir][esq] * self.weights[esq][dir])

    # Esta funcao recebe o array error que guarda todos
    # os valores de erros acumulados para os nos da direita.
    # Recebe, de seguida, o array -  novo_E - cujo valor vai ser actualizado,
    # passando a guardar o valor acumulado de todos os nos da equerda
    def actualiza_novo_E(self, error: np.ndarray, novo_E: np.ndarray):
        for esq in range(len(novo_E)):
            self.actualiza_novo_E_na_posicao(error, novo_E, esq)


class SigmoidActivation:
    """
    A sigmoid activation layer.
    """

    def __init__(self):
        """
        Initialize the sigmoid activation layer.
        """
        pass

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
        return 1 / (1 + np.exp(-X))

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        """
        return error