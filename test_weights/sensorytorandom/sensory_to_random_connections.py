import numpy as np
import matplotlib.pyplot as plt


alpha = 2100

"""
    Rows represent the neurons in the "random" network.
    Columns represent the neurons in the "sensory" network.
"""

def create_weight_matrix_feedforward(sensory_array, random_array, excitatory_probability):
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
        N_excitatory_i = np.count_nonzero(weight_matrix[i] > 0)
        for j in range(N_random):
            if weight_matrix[i, j] > 0:
                weight_matrix[i, j] = (alpha / N_excitatory_i) - (alpha / (8 * N_sensory))
            else:
                weight_matrix[i, j] = - (alpha / (8 * N_sensory))
    
    return weight_matrix

    # we calculate the number of values that are positive
    num_positive = np.sum(weight_matrix > 0)

    print("num_positive: ", num_positive)

    print("alpha: ", alpha)
    print("N_sensory: ", N_sensory)
    w_ex = (alpha / num_positive) - (alpha / (3 * N_sensory))
    w_in = -alpha / (3 * N_sensory)
    
    print("w_ex: ", w_ex)
    print("w_in: ", w_in)  

    # we set all positive values to 1/num_positive
    weight_matrix[weight_matrix > 0] = w_ex

    # we set all negative values to w_in
    weight_matrix[weight_matrix < 0] = w_in
    
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

weight_matrix = create_weight_matrix_feedforward(sensory_array, random_array, excitatory_probability)

#print number of all positive values in matrix:
print("Number of positive values:", np.sum(weight_matrix > 0))

positive_avg, negative_avg = calculate_average(weight_matrix)
print("Average of positive values:", positive_avg)
print("Average of negative values:", negative_avg)