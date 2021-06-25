#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from scipy import optimize as sp
import math
import Set_Emnist_Data as data
from random import uniform


# In[5]:


#calculates sigmoid function, params: z (variable), returns: sigmoid calculation
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

#calculates the sigmoidGradient (derivative), params: z (variable), returns: sigmoid gradient calculation
def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))

def randomly_initialize(num_in, num_out):
    epsilon = 0.12
    size = num_out*(num_in+1)
    all_weights = np.zeros((size,))
    for i in range(size):
        all_weights[i] = uniform(-epsilon, epsilon)
    weights = np.reshape(all_weights,(num_out,num_in+1))
    return weights

#calculates the regularization term for the neural network, params: lambda (regularization constant),
#m (number of training samples), returns regularization value
def regularization(lamda, m):
    lamda_val = lamda/(2.0*m)
    theta1_sum = 0
    theta2_sum = 0
    for j in range(len(data.theta1)-1):
        for k in range(data.theta1[0].size-1):
            theta1_sum += data.theta1[j+1][k+1]*data.theta1[j+1][k+1]
    for j in range(len(data.theta2)-1):
        for k in range(data.theta2[0].size-1):
            theta2_sum += data.theta2[j+1][k+1]*data.theta2[j+1][k+1]
    return lamda_val*(theta1_sum+theta2_sum)

#calculates the cost for the neural network, params: y_vals (expected output values), hyp (calculated output values),
#m (number of training samples), returns cost between given sample and expected value
def calc_cost(y_vals, hyp, lamda, m): #hyp and y are both 10x1 vectors
    cost = 0
    for k in range(y_vals.size):
        cost += -y_vals[k] * math.log(hyp[k]) - (1-y_vals[k])*math.log(1-hyp[k])
    return cost

#predicts the number that correlates to the input data, params: weights(an array that consists of 2 weight matricies),
#x_vals (array that consists of input values), returns prediction number (0-9)
def predict(weights, x_vals):
        #x_vals = np.hstack(([1],x_vals))
        weights1 = weights[0]
        weights2 = weights[1]
        z2 = np.matmul(x_vals,weights1.T)
        a2 = sigmoid(z2)
        a2 = np.hstack(([1], a2))
        z3 = np.matmul(a2,weights2.T)
        a3 = sigmoid(z3)
        max_val = a3[0]
        max_index = 0
        print(a3)
        for i in range(len(a3)):
            if (a3[i] > max_val):
                max_val = a3[i]
                max_index = i
        prediction = max_index+1
        if prediction == 10:
            prediction = 0
        return prediction

#performs forward and backward prop to get a final cost value, J, and 2 gradient weight matricies
#params: nn_params(array that consists of 2 weight matricies for layer 1 and 2 respectively), input_layer_size (number of input units),
#hidden_layer_size (number of hidden units), num_labels (number of output units), x (training samples), y (expected output values), lambda_reg (regularization constant)
#returns cost and an array of weight gradient vectors
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg,m):
    data.theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size, input_layer_size+1))
    data.theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1))

    J = 0;
    Theta1_grad = np.zeros_like(data.theta1)
    Theta2_grad = np.zeros_like(data.theta2)

    #Forward and Back prop:

    bigDelta1 = 0
    bigDelta2 = 0
    cost_temp = 0

    # for each training example
    for t in range(m):

        ## step 1: perform forward pass
        x = X[t]

        #calculate z2 (linear combination) and a2 (activation for layer 2)
        z2 = np.matmul(x,data.theta1.T)
        a2 = sigmoid(z2)

        # add column of ones as bias unit to the second layer
        a2 = np.hstack(([1], a2))
        # calculate z3 (linear combination) and a3 (activation for layer 3 aka final hypothesis)
        z3 = np.matmul(a2,data.theta2.T)
        a3 = sigmoid(z3)

        #Backpropogation:

        #step 2: set delta 3
        delta3 = np.zeros((num_labels))

        #Get Error: subtract actual val in y from each hypothesized val in a3
        y_vals = np.zeros((num_labels))
        for k in range(num_labels): #for each of the 10 labels subtract
            y_k = y[t][k]
            y_vals[k] = y_k
            delta3[k] = a3[k] - y_k

        #step 3: for layer 2 set delta2 = Theta2 Transpose * delta3 .* sigmoidGradient(z2) (= Chain Rule)
        #Skip over the bias unit in layer 2: no gradient calculated for this value
        delta2 = np.matmul(data.theta2[:,1:].T, delta3) * sigmoidGradient(z2)

        #step 4: accumulate gradient from this sample
        bigDelta1 += np.outer(delta2, x)
        bigDelta2 += np.outer(delta3, a2)
        #Update the total cost given the cost from this sample
        cost_temp += calc_cost(y_vals, a3, lambda_reg, data.m)

    #Accumulate cost values and regularize to get Cost(J)
    term1 = (1/data.m)*cost_temp
    term2 = regularization(lambda_reg, data.m)
    J = term1 + term2

    # step 5: obtain gradient for neural net cost function by dividing the accumulated gradients by m
    Theta1_grad = bigDelta1 / data.m
    Theta2_grad = bigDelta2 / data.m


    #Regularization
    #only regularize for j >= 1, so skip the first column
    Theta1_grad_unregularized = np.copy(Theta1_grad)
    Theta2_grad_unregularized = np.copy(Theta2_grad)
    Theta1_grad += (float(lambda_reg)/data.m)*data.theta1
    Theta2_grad += (float(lambda_reg)/data.m)*data.theta2
    Theta1_grad[:,0] = Theta1_grad_unregularized[:,0]
    Theta2_grad[:,0] = Theta2_grad_unregularized[:,0]
    flattened_grads = np.hstack((Theta1_grad.flatten(),Theta2_grad.flatten()))

    return J, flattened_grads
