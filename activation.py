
import numpy as np

def activation_fake_spikes(sum_i):
    r_i = get_baseline_shifted_hyperbolic_tangent(sum_i)
    print(r_i)
    return get_spikes(r_i)

def get_baseline_shifted_hyperbolic_tangent(g):
    return 1/(0.4 * (1 + np.tanh(0.4*g - 3)))

def get_spikes(r_t, max = 10):
    t = 0
    spikes = []
    while t < max:
        next_spike = np.random.poisson(r_t)
        if next_spike + t < max:
            t += next_spike
            spikes.append(t)
        else:
            break
    return spikes

for i in range(10):
    print(get_spikes(i +1,10))
