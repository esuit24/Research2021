import numpy as np
from random import uniform
def randomly_initialize(num_in, num_out):
    epsilon = 0.12
    size = num_out*(num_in+1)
    all_weights = np.zeros((size,))
    for i in range(size):
        all_weights[i] = uniform(-epsilon, epsilon)
    weights = np.reshape(all_weights,(num_out,num_in+1))
    return weights
