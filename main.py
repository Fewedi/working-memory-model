
import numpy as np

# Stepsto run the simulation
STEPS = 1
# Step to stop the initialisation of the input layer
STEP_STOP_INIT = 10
# Amount of neurons in the visual layer
N_SENS = 4
# Amount of rings
SENS_AMOUNT = 1
# Amount of neurons in the random Layer
N_RAND = 2


def setup_visual_layer_neurons():
    #TODO: Implement setup of visual layer neurons
    return np.random.rand(SENS_AMOUNT * N_SENS) * 2 - 1

def setup_random_layer_neurons():
    #TODO: Implement setup of random layer neurons
    return np.random.rand(N_RAND) * 2 - 1

def setup_input_layer_neurons():
    #TODO: Implement setup of input layer neurons
    return np.random.rand(SENS_AMOUNT * N_SENS) * 2 - 1

def setup_weights_ixv():
    # is this sufficient?
    return np.diag(SENS_AMOUNT * N_SENS, SENS_AMOUNT * N_SENS)* 2 - 1
    
def setup_weights_vxv():
    #TODO: Implement setup of weights between visual layer neurons
    return np.random.rand(SENS_AMOUNT * N_SENS, SENS_AMOUNT * N_SENS) * 2 - 1

def setup_weights_vxr():
    #TODO: Implement setup of weights between visual and random layer neurons
    return np.random.rand(SENS_AMOUNT * N_SENS, N_RAND) * 2 - 1

def setup_weights_rxv(vxr):
    #TODO: Implement setup of weights between random and visual layer neurons
    return vxr.transpose()

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
    visual_layer_neurons = setup_visual_layer_neurons()
    random_layer_neurons = setup_random_layer_neurons()
    input_layer_neurons = setup_input_layer_neurons()
    weights_ixv = setup_weights_ixv()
    weights_vxv = setup_weights_vxv()
    weights_vxr = setup_weights_vxr()
    weights_rxv = setup_weights_rxv(weights_vxr)

    for i in range(STEPS):
        print("VISUAL LAYER NEURONS: ")
        print("---------------------------------------")
        new_visual_layer_neurons = activation_function(merge_two_transfers(merge_two_transfers(
                                                                           transfer_function(input_layer_neurons, weights_ixv), 
                                                                           transfer_function(visual_layer_neurons, weights_vxv)), 
                                                                           transfer_function(random_layer_neurons, weights_vxr)),
                                                                            visual_layer_neurons)
        print("RANDOM LAYER NEURONS: ")
        print("---------------------------------------")
        new_random_layer_neurons = activation_function(transfer_function(visual_layer_neurons, weights_rxv), random_layer_neurons)

        visual_layer_neurons = new_visual_layer_neurons
        random_layer_neurons = new_random_layer_neurons

        if i == STEP_STOP_INIT:
            input_layer_neurons = np.zeros(SENS_AMOUNT * N_SENS)

        output[i] = random_layer_neurons
        print("Visual Layer Neurons: ", visual_layer_neurons)
        print("Random Layer Neurons: ", random_layer_neurons)
        print("")
    
    print("OUTPUT: ")
    print("---------------------------------------")
    print(output)

start()