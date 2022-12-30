#10.3 e 10.4 e 10.5
# import numpy as np
from si.data.dataset import Dataset
# nn imports
from si.neural_networks.layer import Dense
from si.neural_networks.activationSigmoid import  SigmoidActivation
from si.neural_networks.ff_nn import FF_NN
from si.neural_networks.activationSoftMax import SoftMaxActivation
from si.neural_networks.activationReLU import ReLUActivation
from si.neural_networks.activationLinear import LinearActivation
import numpy as np
from si.data.dataset import Dataset

#10.3
l1=Dense(input_size=32, output_size=16)
l2=Dense(input_size=16, output_size=1)
#l3=Dense(input_size=1, output_size=1)
#f3_sig=SigmoidActivation()
f1_sig=SigmoidActivation()
f2_sig=SigmoidActivation()
ff_nn=FF_NN(layers=[l1,f1_sig,l2,f2_sig])
#ff_nn.predict(dataset=)

#10.4
l14=Dense(input_size=32, output_size=16)
l24=Dense(input_size=16, output_size=1)
#l3=Dense(input_size=1, output_size=1)
f_sig=SigmoidActivation()
f_soft=SoftMaxActivation()
ff_nn=FF_NN(layers=[l14,f_sig,l24,f_soft])

# NOTE: since we apply sigmoid in the
#output activation layer, then our
#estimation (y_pred) is a value
#from 0 to 1. So, to find the class
#of our input we can just round that value
#(see it as the probability of belonging
#to class=1)

# dataset_ = ...
# y_pred = ff_nn.predict(dataset= dataset_)
# class = y_pred.round()




#10.4 com 1-hot-encoding
l1=Dense(input_size=32, output_size=16)
l2=Dense(input_size=16, output_size=3)
f1_sig=SigmoidActivation()
f2_sm=SoftMaxActivation()
ff_nn=FF_NN(layers=[l1,f1_sig,l2,f2_sm])

# NOTE: we have 3 output nodes => apply argmax to the network
# prediction to find the "estimated" class

# dataset_ = ...
# y_pred = ff_nn.predict(dataset= dataset_)
# class = np.argmax(y_pred)

l24_hot=Dense(input_size=16, output_size=3)
ff_nn_hot=FF_NN(layers=[l14,f_sig,l24_hot,f_soft])

# 10.5
l1=Dense(input_size=32, output_size=16)
l2=Dense(input_size=16, output_size=1)
f1_relu=ReLUActivation()
f2_linear = LinearActivation()
ff_nn=FF_NN(layers=[l14,f1_relu,l24, f2_linear])