#!/usr/bin/env python
# coding: utf-8

# In[14]:


import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt


# In[10]:


# We use keras to encode the network :

the_network = tf.keras.Sequential([
              tf.keras.layers.Dense(3, activation='relu', input_shape=(2,)),     #  the hidden layer with 3 neurons
              tf.keras.layers.Dense(1, activation='sigmoid')                     #  the output layer with 1 neuron
])

the_network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# XOR gate : input and output 
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
output_data = np.array([[0], [1], [1], [0]])             

# Training
the_network.fit(input_data, output_data, epochs=1000, verbose=0)  

# Predicting
predictions = the_network.predict(input_data)
print("Predictions:", predictions)


# In[12]:


# evaluate the model :

loss, accuracy = the_network.evaluate(input_data, output_data, verbose=0)
print(f"Model Loss: {loss:.2f}%")
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# In[16]:


# recording the training history :

history = the_network.fit(input_data, output_data, epochs=1000, verbose=0)

# printing the loss over several epochs

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Loss', color='blue')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# In[17]:


# To extract and print weights and biases

def print_layer_weights(ANNmodel):
    for i, layer in enumerate(ANNmodel.layers):
        
        weights = layer.get_weights()
        
        print(f"Layer {i + 1} weights:\n{weights[0]}")     # Print weight matrix
        
        if len(weights) > 1: 
            print(f"Layer {i + 1} biases:\n{weights[1]}")  # Print bias vector
        print()  

print_layer_weights(the_network)


# In[18]:


# To compute the errors :

errors = (predictions - output_data) ** 2

for i in range(len(predictions)):
    print(f"Input {i + 1}: Computed Output: {predictions[i][0]:.4f}, True Output: {output_data[i][0]}, Error: {errors[i][0]:.4f}")


# In[ ]:




