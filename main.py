
import numpy as np

# Stepsto run the simulation
STEPS = 1
# Step to stop the initialisation of the input layer
STEP_STOP_INIT = 10
# Amount of neurons in the sensory layer
# final should be 512
N_SENS = 16
# Amount of rings Defaulting with one ring atm
# SENS_AMOUNT = 1
# Amount of neurons in the random Layer
# final should be 1024
N_RAND = 32


alpha_intra_connections = 0.28


def setup_sensory_layer_neurons():
    # Neurons in the sensory layer are arranged in a circle, but the array is only a pointer and does not need any specific value. The ordered numbers are only for better understanding.
    return np.arange(N_SENS)

def setup_random_layer_neurons():
    # The array is only a pointer and does not need any specific value. The ordered numbers are only for better understanding. Random layer should be twice the size of the sensory layer. 
    return np.arange(N_RAND)

# TODO ?? Es gibt nur die beiden layer mit neurons
def setup_input_layer_neurons():
    #TODO: Implement setup of input layer neurons
    return np.random.rand(N_SENS) * 2 - 1

def setup_weights_ixs():
    # is this sufficient?
    a = N_SENS * 2 - 1
    b = N_SENS * 2 - 1
    return np.diag(np.random.rand(a), b)
    
def setup_weights_sxs(sensory_neuron_array):
    # Setup of weights between sensory layer neurons
    k1 = 1.0
    k2 = 0.25
    A = 2.0
    num_neurons = len(sensory_neuron_array)
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
                weight = alpha_intra_connections + first_term - second_term
                weight_matrix[i, j] = weight

    return weight_matrix


def setup_weights_sxr(sensory_array, random_array, excitatory_probability): #def create_weight_matrix_feedforward():
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

def setup_weights_rxs(sxr):
    #TODO: Implement setup of weights between random and sensory layer neurons
    return sxr.transpose()

def setup_output_matrix():
    print("Output Matrix: ")
    print(np.zeros((STEPS, N_RAND)))
    return np.zeros((STEPS, N_RAND))

def transfer_function(layer_neurons, weights):
    # is this sufficient?
    print("")
    print("Shape Layer Neurons: ", layer_neurons.shape)
    print("Layer Neurons: ", layer_neurons)
    print("Shape Weights: ", weights.shape)
    print("Weights: ", weights)
    print("shape Result: ", np.dot(weights, layer_neurons).shape)
    print("Result: ", np.dot(weights, layer_neurons))
    
    return np.dot(weights,layer_neurons)

def activation_function(transfer_input, layer_neurons):
    #TODO: Implement activation function
    # sind die layer_neurons überhaupt relevant hierfür?
    
    return transfer_input

def merge_two_transfers(first_sum, second_sum):
    return np.add(first_sum, second_sum)

def start():
    output = setup_output_matrix()
    sensory_layer_neurons = setup_sensory_layer_neurons()
    random_layer_neurons = setup_random_layer_neurons()
    input_layer_neurons = setup_input_layer_neurons()
    weights_ixs = setup_weights_ixs()
    weights_sxs = setup_weights_sxs()
    weights_sxr = setup_weights_sxr()
    weights_rxs = setup_weights_rxs(weights_sxr)

    for i in range(STEPS):
        print("SENSORY LAYER NEURONS: ")
        print("---------------------------------------")
        new_sensory_layer_neurons = activation_function(merge_two_transfers(merge_two_transfers(
                                                                           transfer_function(input_layer_neurons, weights_ixs), 
                                                                           transfer_function(sensory_layer_neurons, weights_sxs)), 
                                                                           transfer_function(random_layer_neurons, weights_sxr)),
                                                                            sensory_layer_neurons)
        print("RANDOM LAYER NEURONS: ")
        print("---------------------------------------")
        new_random_layer_neurons = activation_function(transfer_function(sensory_layer_neurons, weights_rxs), random_layer_neurons)

        sensory_layer_neurons = new_sensory_layer_neurons
        random_layer_neurons = new_random_layer_neurons

        if i == STEP_STOP_INIT:
            input_layer_neurons = np.zeros(SENS_AMOUNT * N_SENS)

        output[i] = random_layer_neurons
        print("Sesnory Layer Neurons: ", sensory_layer_neurons)
        print("Random Layer Neurons: ", random_layer_neurons)
        print("")
    
    print("OUTPUT: ")
    print("---------------------------------------")
    print(output)

start()