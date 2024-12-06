#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Homework # 3
# Problem # 3


# In[3]:


import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
print(tf.__version__)


# In[ ]:


# One way to solve the problem is by using tensorflow library :


# In[5]:


V1 = tf.constant([2.3, 1.2, 0.3, 0.0])
V2 = tf.constant([1.9, 1.7, 2.6, 0.2, 1.3])

V1_softmax = tf.nn.softmax(V1)
V2_softmax = tf.nn.softmax(V2)

print("Softmax vector of V is 1:", V1_softmax.numpy())
print("Softmax of V2:", V2_softmax.numpy())


# In[ ]:


# Another way to solve it is by defining the softmax function :


# In[11]:


import numpy as np

def compute_softmax(x):
    
    exp_x = np.exp(x) 
    sum_exp_x = np.sum(exp_x)
    softmax_values = exp_x / sum_exp_x
    
    return softmax_values

# Define the input vectors
V1 = np.array([2.3, 1.2, 0.3, 0.0])
V2 = np.array([1.9, 1.7, 2.6, 0.2, 1.3])

# Compute the softmax for both vectors
print("Softmax for V1:")
softmax_V1 = compute_softmax(V1)
print(softmax_V1)

print("Softmax for V2:")
softmax_V2 = compute_softmax(V2)
print(softmax_V2)


# In[ ]:





# In[ ]:




