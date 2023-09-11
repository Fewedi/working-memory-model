
import numpy as np
import math

time = 10

# --- Activation - poisson for generating time between spikes ---

def activation_function_f(new_dotted_input):
    fire_rate = (1/time)*(0.4 * (1 + math.tanh(0.4*new_dotted_input - 3)))
    spikes = get_spike_train_f(fire_rate)
    synaptic_activation = get_synaptic_activation_f(spikes)
    print(synaptic_activation)
    return synaptic_activation

def get_spike_train_f(fire_rate):
    # creates a spike train from 0 to time
    t = 0
    spikes = []
    while t < time:
        # generate time until next spike
        next_spike = np.random.poisson(fire_rate)
        # check if next spike is within time
        if next_spike + t < time:
            t += next_spike
            spikes.append(t)
        else:
            break
    return spikes

def get_synaptic_activation_f(spikes, delta = 1):
    activation = 0
    for spike in spikes: 
        activation += delta * spike
    return activation/time

def show_spike_train(r,d):
    for i in range(r):
        fire_rate = (i +1)/d
        spikes = get_spike_train_f(fire_rate)
        synaptic_activation = get_synaptic_activation_f(spikes)
        print("Fire Rate: " + str(fire_rate) + " - Spikes: " + str(spikes) )
        print("Synaptic Activation: " + str(synaptic_activation))
    #mean fire rate = 0.02779434411466435

show_spike_train(20,5)