#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[8]:


import numpy as np

def costFunction(x, y):
    return -np.sqrt(25 - (x - 2)**2 - (y - 3)**2)

# Gradients (with corrected typo)
def gradients(x, y):
    denominator = np.sqrt(25 - (x - 2)**2 - (y - 3)**2)
    # Avoid division by zero
    if denominator == 0:
        return 0, 0
    dzdx = (x - 2) / denominator
    dzdy = (y - 3) / denominator
    return dzdx, dzdy

# Gradient Descent function
def gradient_descent(learning_rate=0.01, epsilon=0.0001, max_steps=10000):
    
    # Starting values (a guess)
    x, y = 0.05, 0.05
    steps = 0
    
    for step in range(max_steps):
        
        steps += 1

        # Calculate gradients
        dzdx, dzdy = gradients(x, y)
        
        # Update x and y
        x_new = x - learning_rate * dzdx
        y_new = y - learning_rate * dzdy
        
        # Check for convergence
        if abs(x_new - x) < epsilon and abs(y_new - y) < epsilon:
            break
            
        # Update the current values
        x, y = x_new, y_new
        
    return x, y, steps

# Gradient descent algorithm
optimum_x, optimum_y, total_steps = gradient_descent()

# Calculate z using the optimal x, y
z = np.sqrt(25 - (optimum_x - 2)**2 - (optimum_y - 3)**2)

print(f"Optimal values: x = {optimum_x}, y = {optimum_y}")
print(f"Total steps towards convergence: {total_steps}")
print(f"The value of z is : {z}")


# In[ ]:




