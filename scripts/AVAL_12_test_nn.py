import numpy as np
from si.data.dataset import Dataset
# nn imports
from si.neural_networks.activationSigmoid import SigmoidActivation
from si.neural_networks.layer import Dense
from si.neural_networks.activationReLU import ReLUActivation
from si.neural_networks.nn import NN
from si.metrics.AVAL_11_cross_entropy import cross_entropy_derivative, cross_entropy
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[1],
              [0],
              [0],
              [1]])

#X = np.array([[0,1]])
#y = np.array([[0]])

dataset = Dataset(X, y, features=['x1', 'x2'], label='X1 XNOR X2')
print(dataset.to_dataframe())

# weights for Dense Layer 1

w1 = np.array([[20, -20],
               [20, -20]])
b1 = np.array([[-30, 10]])

l1 = Dense(input_size=2, output_size=2)
l1.weights = w1
l1.bias = b1

# weights for Dense Layer 2

w2 = np.array([[20, 1, 2],
               [20, 3, 4]])
b2 = np.array([[-10, 0, 3]])

l2 = Dense(input_size=2, output_size=3)
l2.weights = w2
l2.bias = b2

w3 = np.array([[20, 1],
               [20, 3],
               [10, 5]])
b3 = np.array([[-10, 0]])

l3 = Dense(input_size=3, output_size=2)
l3.weights = w3
l3.bias = b3

l1_sg = SigmoidActivation()
l2_relu = ReLUActivation()
l3_sg = SigmoidActivation()


# layers
layers = [
    l1,
    l1_sg,
    #l1_relu,
    l2,
    l3_sg
]

# NN
nn = NN(layers=layers, epochs=10, loss_derivative=cross_entropy_derivative, loss=cross_entropy)
#print("prediction")
#print(nn.predict(dataset=dataset))
nn.fit(dataset)
print(nn.history)




