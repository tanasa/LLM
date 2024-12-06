#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Homework # 3
# Problem # 4


# In[2]:


import math
import tensorflow as tf


# In[ ]:


# the target values are 0


# In[6]:


target_values = tf.constant([0, 0, 0, 0, 0], dtype=tf.float32)
computed_values = tf.constant([0.95, 0.8, 0.6, 0.4, 0.1], dtype=tf.float32)

def cost_function(target, computed):
    cost = - (target * tf.math.log(computed) + 
             (1 - target) * tf.math.log(1 - computed))
    return cost   

# Output the values of the cost function :
cost_values = cost_function(target_values, computed_values).numpy()  # Convert to NumPy array for better readability
for i in range(len(cost_values)):
    print(f"Target: {target_values[i]}, Computed Value: {computed_values[i]:.2f}, Cost Function: {cost_values[i]:.4f}")


# In[ ]:


# the target values are 1


# In[7]:


target_values = tf.constant([1, 1, 1, 1, 1], dtype=tf.float32)
computed_values = tf.constant([0.95, 0.8, 0.6, 0.4, 0.1], dtype=tf.float32)

def cost_function(target, computed):
    cost = - (target * tf.math.log(computed) + 
             (1 - target) * tf.math.log(1 - computed))
    return cost   

# Output the values of the cost function :
cost_values = cost_function(target_values, computed_values).numpy()  # Convert to NumPy array for better readability
for i in range(len(cost_values)):
    print(f"Target: {target_values[i]}, Computed Value: {computed_values[i]:.2f}, Cost Function: {cost_values[i]:.4f}")


# In[ ]:




