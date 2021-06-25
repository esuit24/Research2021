#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:

xfile = '/Users/elliesuit/emnist/Emnist Data/X.csv'
yfile = '/Users/elliesuit/emnist/Emnist Data/Y.csv'
#df = pd.read_csv(r'/Users/elliesuit/emnist/Emnist Data/Theta1.csv', header = None)
#df2 = pd.read_csv(r'/Users/elliesuit/emnist/Emnist Data/Theta2.csv', header = None)
df3 = pd.read_csv(xfile, header = None)
df4 = pd.read_csv(yfile, header = None)
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
num_layers = 2

theta1 = np.zeros([hidden_layer_size,input_layer_size+1])
theta2 = np.zeros([num_labels,hidden_layer_size+1])
x = np.zeros((len(df3),input_layer_size))
y_vec = np.zeros((len(df4),))

index = 0
while(index<len(x)):
    x[index] = df3.iloc[index]
    index+=1
#ones = np.ones((len(df3),1))
#x = np.hstack((ones, x))

index = 0
while (index<len(y_vec)):
    y_vec[index] = df4.iloc[index]
    index+=1

m = len(x)
# set y to be a 2-D matrix with each column being a different sample and each row corresponding to a value 0-9
y = np.zeros((m,num_labels))
# for every label, convert it into vector of 0s and a 1 in the appropriate position
for i in range(m): #each row is new training sample
    index = int(y_vec[i]-1)
    y[i][index] = 1


# In[21]:





# In[ ]:
