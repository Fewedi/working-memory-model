
import numpy as np

STEPS = 1
STEP_STOP_INIT = 10
VIS_LAYER_SIZE = 4
VIS_LAYER_AMOUNT = 1
RAND_LAYER_SIZE = 2

def setup_visual_layer_neurons():
    return np.random.rand(VIS_LAYER_AMOUNT * VIS_LAYER_SIZE) * 2 - 1

def setup_random_layer_neurons():
    return np.random.rand(RAND_LAYER_SIZE) * 2 - 1

def setup_input_layer_neurons():
    return np.random.rand(VIS_LAYER_AMOUNT * VIS_LAYER_SIZE) * 2 - 1

def setup_weights_ixv():
    return np.random.rand(VIS_LAYER_AMOUNT * VIS_LAYER_SIZE, VIS_LAYER_AMOUNT * VIS_LAYER_SIZE)* 2 - 1
    # return np.array([[1.0, 1.0, 1.0, 1.0],
    #                     [1.0, 1.0, 1.0, 1.0],
    #                     [1.0, 1.0, 1.0, 1.0],
    #                     [1.0, 1.0, 1.0, 1.0]])

def setup_weights_vxv():
    return np.random.rand(VIS_LAYER_AMOUNT * VIS_LAYER_SIZE, VIS_LAYER_AMOUNT * VIS_LAYER_SIZE) * 2 - 1
    return np.array([[0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]])

def setup_weights_vxr():
    return np.random.rand(VIS_LAYER_AMOUNT * VIS_LAYER_SIZE, RAND_LAYER_SIZE) * 2 - 1
    return np.array([[-0.3, 0.0, 0.6, 0.0],
                     [0.0, 0.5, 0.4, -0.2],
                        [0.0, 0.0, 0.4, 0.0],
                        [0.1, -0.9, 0.0, 0.3]])

def setup_output_matrix():
    print("Output Matrix: ")
    print(np.zeros((STEPS, RAND_LAYER_SIZE)))
    return np.zeros((STEPS, RAND_LAYER_SIZE))

def transfer_function(layer_neurons, weights):
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
    # sind die layer_neurons überhaupt relevant?
    
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

    for i in range(STEPS):
        print("VISUAL LAYER NEURONS: ")
        print("---------------------------------------")
        new_visual_layer_neurons = activation_function(merge_two_transfers(merge_two_transfers(transfer_function(input_layer_neurons, weights_ixv), 
                                                                           transfer_function(visual_layer_neurons, weights_vxv)), 
                                                                           transfer_function(random_layer_neurons, weights_vxr)),
                                                                            visual_layer_neurons)
        print("RANDOM LAYER NEURONS: ")
        print("---------------------------------------")
        new_random_layer_neurons = activation_function(transfer_function(visual_layer_neurons, weights_vxr.transpose()), random_layer_neurons)

        visual_layer_neurons = new_visual_layer_neurons
        random_layer_neurons = new_random_layer_neurons

        if i == STEP_STOP_INIT:
            input_layer_neurons = np.zeros(VIS_LAYER_AMOUNT * VIS_LAYER_SIZE)

        output[i] = random_layer_neurons
        print("Visual Layer Neurons: ", visual_layer_neurons)
        print("Random Layer Neurons: ", random_layer_neurons)
        print("")
    
    print("OUTPUT: ")
    print("---------------------------------------")
    print(output)



start()