import numpy as np
import matplotlib.pyplot as plt

alpha = 2100

def create_weight_matrix_feedforward(sensory_array, random_array, excitatory_probability):
    num_sensory = len(sensory_array)
    num_random = len(random_array)
    
    weight_matrix = np.zeros((num_random, num_sensory))
    
    for i in range(num_random):
        for j in range(num_sensory):
            if np.random.rand() < excitatory_probability:
                weight_matrix[i, j] = 1  # Excitatory weight range
            else:
                weight_matrix[i, j] = -1  # Inhibitory weight range
    
    # Adjust weights based on your specified formula
    alpha = 1.0  # You need to define the alpha value
    N_sensory = num_sensory
    
    for i in range(num_random):
        N_excitatory_i = np.count_nonzero(weight_matrix[i] > 0)
        for j in range(num_sensory):
            if weight_matrix[i, j] > 0:
                weight_matrix[i, j] = (alpha / N_excitatory_i) - (alpha / (8 * N_sensory))
            else:
                weight_matrix[i, j] = - (alpha / (8 * N_sensory))
    
    return weight_matrix


    # we calculate the number of values that are positive
    num_positive = np.sum(weight_matrix > 0)

    print("num_positive: ", num_positive)

    print("alpha: ", alpha)
    print("num_sensory: ", num_sensory)
    w_ex = (alpha / num_positive) - (alpha / (3 * num_sensory))
    w_in = -alpha / (3 * num_sensory)
    
    print("w_ex: ", w_ex)
    print("w_in: ", w_in)  

    # we set all positive values to 1/num_positive
    weight_matrix[weight_matrix > 0] = w_ex

    # we set all negative values to w_in
    weight_matrix[weight_matrix < 0] = w_in
    
    return weight_matrix

sensory_array = np.arange(16)
random_array = np.arange(32)
excitatory_probability = 0.35

weight_matrix = create_weight_matrix_feedforward(sensory_array, random_array, excitatory_probability)

print(weight_matrix)