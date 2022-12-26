import numpy as np
def random_element(values):
    random_index = np.random.randint(0, high=len(values))
    return values[random_index]


def random_parameter_combination(parameter_distribution):
    results = {}

    for key in parameter_distribution:
        values = parameter_distribution[key]
        results[key] = random_element(values)

    return results

 # parameter distribution
parameter_distribution_ = {}
parameter_distribution_['l2_penalty']= np.linspace(1, 10, 10)
parameter_distribution_['alpha']= np.linspace(0.001, 0.0001, 100)
parameter_distribution_['max_iter']= np.linspace(1000, 2000, 200,dtype=int)

#for i in range(50):
#    random_parameter_combination(parameter_distribution_)


#    print(random_parameter_combination(parameter_distribution_))

for val in parameter_distribution_['max_iter']:
    print(str(val))



self.k_mers = [''.join(k_mer) for k_mer in itertools.product('ACTG', repeat=self.k)]


----
l1 = Dense(input_size=32, output_size=16)

# weights for Dense Layer 2

l2 = SigmoidDense(input_size=16, output_size=1)

# layers
layers = [
    l1,
    l2,
]

# NN
ff_nn = FF_NN(layers=layers)
ff_nn.predict(dataset=X)