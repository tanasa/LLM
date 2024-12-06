#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Homework # 3
# Problem # 5


# In[ ]:


# The TensorFlow ‘argmax’ function returns the index of the maximum number. 


# In[1]:


import tensorflow as tf

a = tf.constant([[5, 2, 3],
                 [26, 56, 92],
                 [3, 0, 26]])

a1 = tf.argmax(a, axis = 0)  # max indices along columns
a2 = tf.argmax(a, axis = 1)  # max indices along rows

print("the argmax values are :")
print("a1:", a1.numpy())
print("a2:", a2.numpy())


# In[ ]:




