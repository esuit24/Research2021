import math
import numpy as np
from random import uniform


# In[33]:


def gaussian(mean, sigma = 1): #need a range of x values to choose from
    #each sample has a bunch of different x values to represent the given Gaussian
    #each gaussian is defined by the mean and the standard deviation so choosing discrete x values to identify
    #the function with can pinpoint different areas on the graph to choose a mean value
    #choose x_vals in range (-1,1)
    samples = np.zeros((21,))
    c = 1/(sigma * math.sqrt(2*math.pi))
    x_val = -1
    index = 0
    while(x_val <= 1):
        exp = -((x_val-mean)**2) / (2*sigma**2)
        samples[index] = c*(math.e**exp)
        index += 1
        x_val += 0.1
    return samples
def create_training_set(num_samples):
    #because x_vals in range (-1,1)... mean must be in range (-0.25, 0.25)
    training_data = np.zeros((num_samples, 21))
    means = np.zeros((num_samples,))
    sigmas = np.zeros((num_samples,))
    sample = 0
    while (sample < len(training_data)):
        random_mean = uniform(-0.25, 0.25)
        random_sigma = uniform(-0.25, 0.25)
        means[sample] = random_mean
        sigmas[sample] = random_sigma
        training_data[sample] = gaussian(random_mean, random_sigma)
        sample += 1
    return training_data, means, sigmas
