import numpy as np

def add_shot_noise(trace_sample, counts):

    discrete_trace = np.round(trace_sample*counts)

    noisy_trace = np.random.poisson(lam=discrete_trace)

    return noisy_trace
