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

neuron_array = np.arange(32)

weight_matrix = create_weight_matrix(neuron_array)
for i in range(32):
    print(f'{weight_matrix[5][i]} ,')