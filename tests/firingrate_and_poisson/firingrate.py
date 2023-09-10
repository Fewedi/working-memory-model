
import math
import numpy as np
import random



def firing_rate_function(input):
    sum = np.sum(input)
    spiking_rate = 0.4 * (1 + math.tanh((0.4 * sum) - 3))

    return spiking_rate

# Define the size of the array
array_size = 1024  # Change this to your desired array size

# Generate a random array of the specified size with values between 0 and 1
random_array = [random.random() for _ in range(array_size)]
firing_rate = firing_rate_function(random_array)

print(firing_rate)

import numpy as np

# Generate an array of length 10 with Poisson-distributed values
poisson_array = np.random.poisson(firing_rate, size=10)

print(poisson_array)

previous_activation = random.random()

activation = np.sum(1 - poisson_array) - (previous_activation / len(poisson_array))
if activation < 0:
    activation = 0
print(activation)