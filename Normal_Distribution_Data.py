import math
import numpy as np
from random import uniform
from scipy.integrate import quad
from scipy.special import gamma

#assume domain is (-1,1)

def function(location, scale, shape, num_x_vals): #need a range of x values to choose from
    curr_x = -5
    index = 0
    vals = np.zeros((num_x_vals,))
    increment = 10.0/(num_x_vals-1)
    while(curr_x <= 5):
        c = shape/(2*scale*gamma(curr_x)*(1/shape))
        exp_val = (abs(curr_x-location)/scale)**shape
        ex = -exp_val
        vals[index] = c* math.e ** (ex)
        curr_x += increment
        index += 1
    return vals

def create_training_set(num_samples, num_x_vals = 21): #pass in number of x values as parameter instead of hard coding, accuracy varies with input?
    #because x_vals in range (-1,1)... mean must be in range (-0.25, 0.25)
    training_data = np.zeros((num_samples, num_x_vals))
    mu_vals = np.zeros((num_samples,))
    alpha_vals = np.zeros((num_samples,))
    beta_vals = np.zeros((num_samples,))
    for i in range(num_samples):
        #initializing labels
        #print("sample number: "  + str(i))
        random_mu = uniform(-5, 5)
        random_mu = 0 #remove later
        mu_vals[i] = random_mu
        random_alpha = uniform(0,5)
        random_alpha = 1 #remove later
        alpha_vals[i] = random_alpha
        random_beta = uniform(1,5) #(0,1) (not inclusive of 0) is in the domain too but change later to accomodate for complex
        beta_vals[i] = random_beta
        training_data[i] = function(random_mu, random_alpha, random_beta, num_x_vals)

    return training_data, mu_vals, alpha_vals, beta_vals
