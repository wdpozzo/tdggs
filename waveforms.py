import numpy as np
from numba import njit

@njit
def sinusoid(t, p):
    """
    simple sine wave
    """
    return np.exp(p[0])*np.cos(2.*np.pi*(np.exp(p[1])*t+0.5*np.exp(p[2])*t**2)+p[3])

@njit
def logfdot(logf, a, b):
    """
    linear relation between f and fdot
    """
    return a+b*logf

@njit
def burst(t, f):
    """
    simple sine gaussian
    """
    return np.exp(f[0])*np.cos(2.*np.pi*np.exp(f[1])*t)*np.exp(-(t-f[2])/f[3])**2
