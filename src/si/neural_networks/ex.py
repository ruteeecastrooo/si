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
w4 = np.array([[20],
               [-20],
               [-30]])
b4 = np.array([-30])
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