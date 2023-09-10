
import math
import numpy as np
import matplotlib.pyplot as plt

def create_weight_matrix(neuron_array):
    k1 = 1.0
    k2 = 0.25
    A = 2.0
    alpha = 0.28
    num_neurons = len(neuron_array)
    weight_matrix = np.zeros((num_neurons, num_neurons))
    
    for i in range(num_neurons):
        for j in range(num_neurons):
            if i == j:
                weight_matrix[i, j] = 0.0  # Self-excitation is set to 0
            else:
                angle_i = 2 * np.pi * i / num_neurons
                angle_j = 2 * np.pi * j / num_neurons
                angle_diff = angle_i - angle_j
                first_term = A * np.exp(k1 * (np.cos(angle_diff) - 1))
                second_term = A * np.exp(k2 * (np.cos(angle_diff) - 1))
                weight = alpha + first_term - second_term
                weight_matrix[i, j] = weight

    return weight_matrix

def firing_rate_function(input):
    spiking_rate = 0.4 * (1 + math.tanh((0.4 * input)))

    return spiking_rate

def neuron_activation(input, previous_activation):
    firing_rate = firing_rate_function(input)

    # Generate an array of length 10 with Poisson-distributed values
    poisson_array = np.random.poisson(firing_rate, size=10)
    activation = np.sum(poisson_array) - (previous_activation / len(poisson_array))
    
    if activation < 0:
        activation = 0
    return (activation)

def activation_function(previous_input, new_dotted_input):
    new_activation = []
    for i in range(len(new_dotted_input)):
        new_activation.append(neuron_activation(new_dotted_input[i], previous_input[i]))
    return new_activation

def create_input(array_size, center, width):
    # Create an array of indices
    indices = np.arange(array_size)

    # Calculate the bell curve values using a Gaussian function
    bell_curve = np.exp(-(np.minimum(np.abs(indices - center), array_size - np.abs(indices - center)) ** 2) / (2 * width ** 2))

    # Normalize the bell curve values to have a maximum value of 1
    bell_curve /= np.max(bell_curve)
    
    return bell_curve

neuron_array = create_input(64, 32, 64/8)

weight_matrix = create_weight_matrix(neuron_array)

activation = activation_function(neuron_array, np.dot(neuron_array, weight_matrix))
print(activation)

activation_2 = activation_function(activation, np.dot(activation, weight_matrix))

indices = np.arange(len(neuron_array))
# Create a figure and axis for the plot
plt.figure(figsize=(8, 4))
plt.title("Array Before and After Activation")
plt.xlabel("Index")
plt.ylabel("Value")

# Plot the data before activation in blue
plt.plot(indices, neuron_array, label="Before Activation", marker='o', linestyle='-')

# Plot the data after activation in red
plt.plot(indices, activation, label="After Activation", marker='x', linestyle='--')

# Plot the data after activation in red
plt.plot(indices, activation_2, label="After After Activation", marker='o', linestyle='-.')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()