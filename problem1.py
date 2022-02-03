""" 
Jackson Arnold 
EEE6504 HW1: Wiener Filter and LMS
Problem 1

Contributors/sources:


Plant TF: 
H(z) = (1-z^-10)/(1-z^-1)

Gaussian noise power N=0.1

alpha stable noise: 
phi(t) = exp(-gamma * abs(t) ^ alpha) 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
from scipy.linalg import circulant

# numpy.correlate(a, v, mode='valid')
# numpy.convolve(a, v, mode='full')

"""
Input X shape = window rows x order columns 

R = X.T @ X
P = X.T @d.T
Wopt = R^-1
"""

order = [5,15,30]
window = [100,500,1000]

samples = 10000
t = np.arange(samples)
gaussian_noise = np.random.normal(0,0.1,samples)
alpha, beta, gamma = 1.8, 0, 1
input_noise = levy_stable.rvs(alpha, beta, size=samples)

w_true = np.ones(10)
print(f"{gaussian_noise.shape = }")

output_noise = np.convolve(np.ravel(input_noise),w_true)[9:10010] + gaussian_noise

for o in order: 
    for w in window: 
        X = np.zeros([w,o])
        d = output_noise[:w]
        for i in range(w):
            X[i,:] = np.ravel(input_noise[i:o+i])

        R = X.T @ X
        wopt = np.linalg.inv(R) @ X.T @ d.T
        print(f"for a {o} order filter window size {w}")
        print(f"{wopt = }")


        
