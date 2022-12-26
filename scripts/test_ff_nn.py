#10.3 e 10.4 e 10.5
# import numpy as np
from si.data.dataset import Dataset
# nn imports
from si.neural_networks.layer import Dense
from si.neural_networks.layerSigmoid import SigmoidDense
from si.neural_networks.ff_nn import FF_NN
from si.neural_networks.RoundActivation import RoundActivation
from si.neural_networks.layerArgmax import ArgmaxDense
from si.neural_networks.layerReLU import ReLUDense
import numpy as np
from si.data.dataset import Dataset

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[1],
              [0],
              [0],
              [1]])

dataset = Dataset(X, y, features=['x1', 'x2'], label='X1 XNOR X2')
dataset.to_dataframe()
#print(dataset.to_dataframe())

# weights for Dense Layer 1

w1 = np.array([[20, -20],
               [20, -20]])
b1 = np.array([[-30, 10]])

l1 = SigmoidDense(input_size=2, output_size=2)
l1.weights = w1
l1.bias = b1

w2 = np.array([[20, -20, -30],
               [20, -20, 15]])
b2 = np.array([[-30, 10, -40]])

l2 = SigmoidDense(input_size=2, output_size=3)
l2.weights = w2
l2.bias = b2

l3= RoundActivation()
l4=ArgmaxDense(input_size=3, output_size=1)
l5=ReLUDense(input_size=3, output_size=1)
# layers
layers = [
    l1,
    l2,
    l3,
    #l4
    #l5
]
# NN
ff_nn = FF_NN(layers=layers)
y=ff_nn.predict(dataset=dataset)
print(y)
#print(ff_nn.score(dataset=dataset))
