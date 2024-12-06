#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Homework # 3
# Problem # 2


# In[10]:


import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
print(tf.__version__)


# In[16]:


def sigmoid_gradient(x):
    """Computes the gradient of the sigmoid function."""
    sig_value = sigmoid(x)                   
    return sig_value * (1 - sig_value)           

def tanh_gradient(x):
    """Computes the gradient of the tanh function."""
    tanh_value = np.tanh(x)  
    return 1 - tanh_value ** 2  

def relu_gradient(x):
    """Computes the gradient of the ReLU function."""

    gradient = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] > 0:
            gradient[i] = 1 
        else:
            gradient[i] = 0
    return gradient

if __name__ == "__main__":

    x = np.array([-4.0, 0.5, 4.0])
    gradient_values_sigmoid = sigmoid_gradient(x)
    print("Gradient values of the sigmoid function are:", gradient_values_sigmoid)

    gradient_values_tanh = tanh_gradient(x)
    print("Gradient values of the tanh function are:", gradient_values_tanh)

    gradient_values_relu = relu_gradient(x)
    print("Gradient values of the relu function are:", gradient_values_relu)


# In[ ]:





# In[ ]:





# In[ ]:




