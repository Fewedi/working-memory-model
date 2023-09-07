import numpy as np
import matplotlib.pyplot as plt


alpha = 2100

"""
    Rows represent the neurons in the "random" network.
    Columns represent the neurons in the "sensory" network.
"""

def sum_excitatory_connections_to_neuron_i(intra_weight_matrix, neuron_i):
    num_neurons = len(intra_weight_matrix)
    excitatory_sum = 0
    
    for j in range(num_neurons):
        if j != neuron_i and intra_weight_matrix[j, neuron_i] > 0:
            excitatory_sum = excitatory_sum + 1
    
    return excitatory_sum

def sum_inter_network_excitatory_connections_to_neuron_i(weight_matrix_feedforward, neuron_i):
    excitatory_sum = np.sum(weight_matrix_feedforward[neuron_i, weight_matrix_feedforward[neuron_i, :] > 0])
    return excitatory_sum

def create_weight_matrix_feedforward(sensory_array, random_array, excitatory_probability, intra_weight_matrix):
    N_sensory = len(sensory_array)
    N_random = len(random_array)
    
    weight_matrix = np.zeros((N_sensory, N_random))  # Swap dimensions
    
    for i in range(N_sensory):  # Loop through sensory neurons
        for j in range(N_random):  # Loop through random neurons
            if np.random.rand() < excitatory_probability:
                weight_matrix[i, j] = 1  # Excitatory weight range
            else:
                weight_matrix[i, j] = -1  # Inhibitory weight range
    
    # Adjust weights based on your specified formula
    alpha = 2100  # You need to define the alpha value
    
    for i in range(N_sensory):
        N_excitatory_i = sum_inter_network_excitatory_connections_to_neuron_i(weight_matrix, i)
        print(N_excitatory_i)
        for j in range(N_random):
            if weight_matrix[i, j] > 0:
                weight_matrix[i, j] = (alpha / N_excitatory_i) - (alpha / (8 * N_sensory))
            else:
                weight_matrix[i, j] = - (alpha / (8 * N_sensory))
    
    return weight_matrix

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

def calculate_average(matrix):
    positive_sum = 0
    positive_count = 0
    negative_sum = 0
    negative_count = 0

    for row in matrix:
        for value in row:
            if value > 0:
                positive_sum += value
                positive_count += 1
            elif value < 0:
                negative_sum += value
                negative_count += 1

    if positive_count > 0:
        positive_avg = positive_sum / positive_count
    else:
        positive_avg = 0

    if negative_count > 0:
        negative_avg = negative_sum / negative_count
    else:
        negative_avg = 0

    return positive_avg, negative_avg

sensory_array = np.arange(512)
random_array = np.arange(1024)
excitatory_probability = 0.35

intra_weight_matrix = create_weight_matrix(sensory_array)

weight_matrix = create_weight_matrix_feedforward(sensory_array, random_array, excitatory_probability, intra_weight_matrix)

#print number of all positive values in matrix:
print("Number of positive values:", np.sum(weight_matrix > 0))

positive_avg, negative_avg = calculate_average(weight_matrix)
print("Average of positive values:", positive_avg)
print("Average of negative values:", negative_avg)
