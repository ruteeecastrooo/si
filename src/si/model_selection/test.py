import numpy as np

def random_element(values):
    random_index= np.random.randint(0, high=len(values))
    return values[random_index]

def random_parameter_combination(parameter_distribution):
    values=[]
    resultados= {}
    resultados.append(random_element(values))
    values1=parameter_distribution['l2_penalty']
    resultados['l2_penalty']=(random_element(values))

 # parameter distribution
 parameter_distribution_ = {
    'l2_penalty': np.linspace(1, 10, 10),
    'alpha':  np.linspace(0.001, 0.0001, 100),
    'max_iter': np.linspace(1000, 2000, 200),

 }