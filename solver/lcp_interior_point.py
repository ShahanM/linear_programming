import numpy as np


def solve(tau, epsilon, theta, x0, s0):
    mu0 = np.dot(x0.T, s0) / n
    v0 = np.sqrt(np.dot(x0, s0)/mu0)

    x = x0
    s = s0
    mu = mu0

    while n*mu >= epsilon:
        mu = (1 - theta)*mu
        v = np.sqrt(np.dot(x, s)/mu)

        while sum([get_kernel(vi) for  vi in v] > tau:




def get_kernel(t):
    return ((t**2 - 1) / 2) - np.log(t)