#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os
import numpy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
print(tf.__version__)


# In[40]:


# Homework # 3
# Problem # 1


# In[42]:


# We use as an argument a string that encodes the name of the function : "sigmoid", "linear", "tanh", "relu"

import tensorflow as tf

def ann(input_data, activation='linear'):    # the default value of the parameter "activation" is "sigmoid"

    try:
        activation_function = getattr(tf.nn, activation) if activation != 'linear' else lambda x: x
    except AttributeError:
        raise ValueError("Invalid activation function. Choose from 'sigmoid', 'linear', 'tanh', or 'relu'.")
    
    # tf.nn: This is TensorFlow's module that contains various neural network functions
    # if activation is set to 'sigmoid', then getattr(tf.nn, 'sigmoid') is equivalent to calling tf.nn.sigmoid.
    
    # Define Layer 1
    W1 = tf.constant([[0.15], [0.05]], dtype=tf.float32)
    b1 = tf.constant([[0.33]], dtype=tf.float32)
    
    # Compute output for Layer 1
    outputH1 = tf.matmul(input_data, W1) + b1
    outputH1_Activation = activation_function(outputH1)
    a1 = outputH1_Activation.numpy()[0, 0]
    
    # Define Layer 2
    W2 = tf.constant([[0.36]], dtype=tf.float32)
    b2 = tf.constant([[0.56]], dtype=tf.float32)
    
    # Compute output for Layer 2
    outputH2 = tf.matmul(outputH1_Activation, W2) + b2
    outputH2_Activation = activation_function(outputH2)
    a2 = outputH2_Activation.numpy()[0,0]
    
    return (a1, a2)

# Input data :

input_data = tf.constant([[0.1, 0.2]], dtype=tf.float32)

# The output of the network for distinct activation functions :

a1, a2 = ann(input_data, activation='sigmoid')
print(f"sigmoid : a1: {a1:.3f} a2: {a2:.3f}")

a1, a2 = ann(input_data, activation='linear')
print(f"linear : a1: {a1:.3f} a2: {a2:.3f}")

a1, a2 = ann(input_data, activation='tanh')
print(f"tanh : a1: {a1:.3f} a2: {a2:.3f}")

a1, a2 = ann(input_data, activation='relu')
print(f"relu : a1: {a1:.3f} a2: {a2:.3f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




