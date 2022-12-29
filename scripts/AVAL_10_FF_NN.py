#10.3 e 10.4 e 10.5
# import numpy as np
from si.data.dataset import Dataset
# nn imports
from si.neural_networks.layer import Dense, SigmoidActivation
from si.neural_networks.layerSigmoid import SigmoidDense
from si.neural_networks.ff_nn import FF_NN
from si.neural_networks.RoundActivation import RoundActivation
from si.neural_networks.layerArgmax import ArgmaxDense
from si.neural_networks.layerReLU import ReLUDense
from si.neural_networks.layerSoftMax import SoftMaxActivation
from si.neural_networks.layerReLU import ReLUActivation
import numpy as np
from si.data.dataset import Dataset

#10.3
l1=Dense(input_size=32, output_size=16)
l2=Dense(input_size=16, output_size=1)
#l3=Dense(input_size=1, output_size=1)
f_sig=SigmoidActivation()
ff_nn=FF_NN(layers=[l1,f_sig,l2,f_sig])

#10.4
l14=Dense(input_size=32, output_size=16)
l24=Dense(input_size=16, output_size=1)
#l3=Dense(input_size=1, output_size=1)
f_sig=SigmoidActivation()
f_soft=SoftMaxActivation()
ff_nn=FF_NN(layers=[l14,f_sig,l24,f_soft])

#10.4 com 1-hot-encoding
l24_hot=Dense(input_size=16, output_size=3)
ff_nn_hot=FF_NN(layers=[l14,f_sig,l24_hot,f_soft])

#10.5
l15=Dense(input_size=32, output_size=16)
l25=Dense(input_size=16, output_size=1)
#l3=Dense(input_size=1, output_size=1)
f_relu=ReLUActivation()
#linear activation = identity (outputs the input)
ff_nn=FF_NN(layers=[l14,f_relu,l24])