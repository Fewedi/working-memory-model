import numpy as np

# desired_mean = 0.95
# num_weights = 1000  # Number of weights you want to generate

# # Calculate the standard deviation for the normal distribution.
# # This will depend on how "spread out" you want the weights to be.
# # You can adjust this value based on your preference.
# desired_std_dev = 0.2

# # Generate weights from a normal distribution with the desired mean and standard deviation.
# weights = np.random.normal(desired_mean, desired_std_dev, num_weights)

# # Adjust the generated weights so that their average matches the desired_mean exactly.
# weights_adjusted = weights + (desired_mean - np.mean(weights))

# # Print the statistics of the generated weights.
# print("Generated weights:")
# print(weights_adjusted)
# print("Mean:", np.mean(weights_adjusted))
# print("Standard Deviation:", np.std(weights_adjusted))

# # You can use the weights_adjusted array for further processing.


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
print(weight_matrix)
positive_avg, negative_avg = calculate_average(weight_matrix)
print("Average of positive values:", positive_avg)
print("Average of negative values:", negative_avg)