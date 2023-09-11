import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd

# --- used in main_annemarie.py ---


# Stepsto run the simulation
STEPS = 10
# Step to stop the initialisation of the input layer
STEP_STOP_INIT = 2
# Amount of neurons in the sensory layer
# final should be 512
N_SENS = 512
# Amount of rings Defaulting with one ring atm
# SENS_AMOUNT = 1
# Amount of neurons in the random Layer
# final should be 1024
N_RAND = 1024


alpha_intra_connections = 0.28
excitatory_probability = 0.35

def setup_neurons(size):
    return np.zeros(size)

def setup_sensory_layer_neurons():
    # Neurons in the sensory layer are arranged in a circle, but the array is only a pointer and does not need any specific value. The ordered numbers are only for better understanding.
    return np.arange(N_SENS)

def setup_random_layer_neurons():
    # The array is only a pointer and does not need any specific value. The ordered numbers are only for better understanding. Random layer should be twice the size of the sensory layer. 
    return np.arange(N_RAND)

def create_input(input_size, center, width):
    # Create an array of indices
    indices = np.arange(input_size)

    # Calculate the bell curve values using a Gaussian function
    bell_curve = np.exp(-(np.minimum(np.abs(indices - center), input_size - np.abs(indices - center)) ** 2) / (2 * width ** 2))

    # Normalize the bell curve values to have a maximum value of 1
    bell_curve /= np.max(bell_curve)
    
    return bell_curve

def setup_weights_ixs():
    # is this sufficient?
    a = N_SENS * 2 - 1
    b = N_SENS * 2 - 1
    return np.diag(np.random.rand(a), b)

def activation_function(new_dotted_input):
    new_activation = []
    for i in range(len(new_dotted_input)):
        new_activation.append(neuron_activation(new_dotted_input[i]))
    #new_activation = normalize(new_activation, 0, 1)
    new_activation /= np.max(new_activation)
    return new_activation

def neuron_activation(input):
    firing_rate = 0.4 * (1 + math.tanh((0.4 * input -3)))

    # Generate an array of length 10 with Poisson-distributed values
    poisson_array = np.random.poisson(firing_rate, size=10)
    activation = np.sum(poisson_array) - (input / len(poisson_array))
    #activation = firing_rate - (input / 10)
    
    if activation < 0:
        activation = 0
    return activation


# formeln ab Seite 16
# alpha und beta auf seite 20

def setup_weights_sxs(sensory_neuron_array):
    # Setup of weights between sensory layer neurons
    k1 = 1.0
    k2 = 0.25
    a = 2.0
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
                first_term = a * np.exp(k1 * (np.cos(angle_diff) - 1))
                second_term = a * np.exp(k2 * (np.cos(angle_diff) - 1))
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
                desired_std_dev = 0.05  # Define your desired standard deviation here
                weight_matrix[i, j] = np.random.normal(desired_mean, desired_std_dev)
            else:
                weight_matrix[i, j] = - (alpha / (8 * N_sensory))
    
    return weight_matrix

def setup_weights_rxs(feedforward_matrix):
    transposed_matrix = np.transpose(feedforward_matrix)
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

def add_together(first_array, second_array):
    return np.add(first_array, second_array)


def run_simulation(shift = 0):
    input = create_input(N_SENS, N_SENS/2, N_SENS/8)
    sensory_layer = setup_neurons(N_SENS)
    random_layer = setup_neurons(N_RAND)


    inter_sensory_weight_matrix = setup_weights_sxs(sensory_layer)
    feedforward_weight_matrix = setup_weights_sxr(sensory_layer, random_layer, excitatory_probability)
    feedback_weight_matrix = setup_weights_rxs(feedforward_weight_matrix)

    activation = setup_neurons(N_SENS)
    
    dif_list = []
    last_activation_sum = 0.0
    current_difference = 0.0

    values = []
    values.append(input)
    for step in range(STEPS):
        if step < STEP_STOP_INIT:
            activation = add_together(activation, input)
        
        # intra sensory connection:
        sxs_weight_times_input = np.dot(activation, inter_sensory_weight_matrix)
        activation_from_sxs = activation_function_f2(sxs_weight_times_input, shift)
        #activation_from_sxs = activation_function(sxs_weight_times_input)

        sxr_weight_times_input = np.dot(activation_from_sxs, feedforward_weight_matrix)
        activation_from_sxr = activation_function_f2(sxr_weight_times_input, shift)
        #activation_from_sxr = activation_function(sxr_weight_times_input)
        rxs_weight_times_input = np.dot(activation_from_sxr, feedback_weight_matrix)
        activation_from_rxs = activation_function_f2(rxs_weight_times_input, shift)
        #activation_from_rxs = activation_function(rxs_weight_times_input)

        activation = activation_from_rxs
        print (sxr_weight_times_input)
        values.append(activation)

        # for optimising of shift
        #new_activation_sum = np.sum(activation_from_sxs)
        #current_difference = new_activation_sum - last_activation_sum
        #if step > STEP_STOP_INIT:
        #    dif_list.append(current_difference)
        #    print("current_difference: " + str(current_difference))
        #last_activation_sum = new_activation_sum
    
    for i in range(len(values)-N_SENS):
        dif_list.append(np.sum(values[i+N_SENS])-np.sum(values[i+N_SENS-1]))

    print("dif_list: " + str(dif_list))
    
    return values, np.sum(dif_list)


# --- Optimising shift ---

def shift_optimisation():
    last_difference = 1000000
    shift = 0
    shift_list = []
    difference_list = []
    for i in range(3):
        result, difference = run_simulation(shift)
        if difference < last_difference:
            shift += 0.1
        elif difference >= last_difference:
            shift -= 0.1
        
        last_difference = difference
        shift_list.append(shift)
        difference_list.append(difference)
    
    print("shift_list: " + str(shift_list))
    print("difference_list: " + str(difference_list))

# --- Activation - just use the (shifted) firing rate ---

def activation_function_f2(new_dotted_input, shift = 0):
    fire_rate = np.zeros(len(new_dotted_input))
    for i in range(len(new_dotted_input)):
        fire_rate[i] = (4 * (1 + np.tanh(0.4*(new_dotted_input[i] - shift) - 3)))    
    return fire_rate

def show_activation(r,d):
    for i in range(r):
        fire_rate = (i +1)/d
        print("Fire Rate: " + str(fire_rate) + " - Activation: " + str(activation_function_f2(fire_rate)))
    #mean fire rate = 0.02779434411466435


shift_optimisation()