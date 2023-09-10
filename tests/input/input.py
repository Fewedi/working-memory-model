import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
array_size = 64  # Size of the array
center = 18      # Center of the bell curve (position)
width = array_size/8        # Width of the bell curve (standard deviation)

def create_input(array_size, center, width):
    # Create an array of indices
    indices = np.arange(array_size)

    # Calculate the bell curve values using a Gaussian function
    bell_curve = np.exp(-(np.minimum(np.abs(indices - center), array_size - np.abs(indices - center)) ** 2) / (2 * width ** 2))

    # Normalize the bell curve values to have a maximum value of 1
    bell_curve /= np.max(bell_curve)
    
    return bell_curve

print(create_input(array_size, center, width))