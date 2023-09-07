from copy import deepcopy
import numpy as np

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
        for j in range(N_random):
            if weight_matrix[i, j] > 0:
                desired_mean = 0.95  # Define your desired mean here
                desired_std_dev = 0.2  # Define your desired standard deviation here
                weight_matrix[i, j] = np.random.normal(desired_mean, desired_std_dev)
            else:
                weight_matrix[i, j] = - (alpha / (8 * N_sensory))
    
    return weight_matrix

def create_weight_matrix_feedback(feedback_matrix):
    transposed_matrix = np.transpose(feedback_matrix)
    N_random = len(transposed_matrix)
    N_sensory = len(transposed_matrix[0])
    
    # Adjust weights based on your specified formula
    beta = 200  # You need to define the alpha value
    
    for i in range(N_random):
        for j in range(N_sensory):
            if transposed_matrix[i, j] > 0:
                desired_mean = 0.36  # Define your desired mean here
                desired_std_dev = 0.05  # Define your desired standard deviation here
                transposed_matrix[i, j] = np.random.normal(desired_mean, desired_std_dev)
            else:
                transposed_matrix[i, j] = - (beta / N_random)
    
    return transposed_matrix

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

weight_matrix_ff = create_weight_matrix_feedforward(sensory_array, random_array, excitatory_probability)
weight_matrix = create_weight_matrix_feedback(weight_matrix_ff, random_array)

for row in weight_matrix:
    matrix_sum = np.sum(row)
    print("Sum of all values in the matrix:", matrix_sum)

#print number of all positive values in matrix:
print("Number of positive values:", np.sum(weight_matrix > 0))
print(weight_matrix)
positive_avg, negative_avg = calculate_average(weight_matrix)
print("Average of positive values:", positive_avg)
print("Average of negative values:", negative_avg)