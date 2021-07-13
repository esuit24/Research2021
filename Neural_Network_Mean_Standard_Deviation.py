#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import Set_Gaussian_Data as data
import matplotlib.pyplot as plt
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[2]:


#load input data for gaussian
training = data.create_training_set(1000, 501) #outputs an array of x_vals, mean values, and sigma values respectively
x_train = training[0]
y_train = training[1]
z_train = training[2]
y_train = np.vstack((y_train, z_train)) #compile training mean and standard deviation into single data structure
y_train = y_train.T #first column as mean, second column as sd

testing = data.create_training_set(300, 501)
x_test = testing[0]
y_test = testing[1]
z_test = testing[2]
y_test = np.vstack((y_test, z_test)) #compile test mean and standard deviation into single data structure
y_test = y_test.T


# In[3]:


#set layers
model = tf.keras.models.Sequential() #feed forward
#input layer: 21 different x values to represent the gaussian function
model.add(tf.keras.layers.Dense(21, activation = tf.nn.relu))
#hidden layers (2):
model.add(tf.keras.layers.Dense(125, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(125, activation = tf.nn.relu))
#output layer:
model.add(tf.keras.layers.Dense(2,))


# In[ ]:


#optimize
model.compile(optimizer = 'adam',
              loss = 'mean_squared_error',
              metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 81) #90 epochs minimizes the difference between test and training and testing cost


# In[6]:


#evaluate the test set and find test loss and accuracy
results = model.evaluate(x = x_test, y = y_test)
val_loss = results[0] #loss between model and actual
val_acc = results[1] #accuracy of the model based on actual
print(val_loss, val_acc)


# In[7]:


#saves the model data
model.save('mean_and_sd.model')
new_model = tf.keras.models.load_model('mean_and_sd.model')


# In[8]:


#makes predictions for all elements in x_test array
predictions = new_model.predict(x_test) #predicts all means and standard deviations of the x_test dataset
print("X Input Values: " + str(x_test[234]))
print("Actual Mean and SD Values: " +  str(y_test[234]))
print("Predicted Mean and SD Values: " + str(predictions[234]))


# In[9]:


predictions = predictions.T
y_test = y_test.T
plt.scatter(predictions[0],y_test[0]) #mean
plt.title('Predicted Mean vs. Actual Mean')
plt.xlabel('Actual Mean')
plt.ylabel('Predicted Mean')
plt.show()


# In[10]:


plt.scatter(predictions[1],y_test[1]) #SD
plt.title('Predicted SD vs. Actual SD')
plt.xlabel('Actual SD')
plt.ylabel('Predicted SD')
plt.show()


# In[11]:


#percent error
plt.scatter(range(len(x_test)), (predictions[0]-y_test[0])/(y_test[0]))
plt.title('Mean Percent Error')
plt.xlabel('Index')
plt.ylabel('Percent Error')
plt.show()
plt.scatter(range(len(x_test)), (predictions[1]-y_test[1])/(y_test[1]))
plt.title('SD Percent Error')
plt.xlabel('Index')
plt.ylabel('Percent Error')
plt.show()


# In[12]:


plt.hist((predictions[0]-y_test[0])/y_test[0])
plt.title('Distribution of Mean Errors')
plt.ylabel('Number of Occurences')
plt.xlabel('Percent Error')
plt.show()
plt.hist((predictions[1]-y_test[1])/y_test[1])
plt.title('Distribution of SD Errors')
plt.ylabel('Number of Occurences')
plt.xlabel('Percent Error')
plt.show()


# In[ ]:
