import numpy as np
import math

def compute_snr(h,sigma,T):
    return np.sqrt(np.sum(h**2/sigma**2)/T)

# mutation operator
def mutation(genome, r_mut, rng):
    """
    apply a random mutation to a genome
    """
    bitstring = genome.copy()
    g = rng.choice([True, False], len(bitstring), p=[r_mut,1-r_mut])
    for i in range(len(bitstring)):
        if g[i] == True:
            bitstring[i] = 1-bitstring[i]
    return bitstring

def crossover(x, y, rng):
    """
    cross the genome of two parents to produce a child
    """
    idx = rng.choice([True, False], len(x))
    z   = np.where(idx, x, y)
    return z

def wander(x, rng):
    """
    """
    return x + rng.normal(0,1e-5,size = len(x))

def survive(x, rng):
    """
    """
    return x
    
def calculate_required_bits(precision, max_value):
    """
    Calculate the number of bits required to represent a value with a given precision.

    :param precision: The desired precision.
    :param max_value: The maximum value to represent.
    :return: The number of bits required.
    """
    if precision <= 0:
        raise ValueError("Precision must be greater than 0.")
    if max_value <= 0:
        raise ValueError("Max value must be greater than 0.")

    # Calculate the number of bits required
    required_bits = math.ceil(math.log2(max_value / precision))

    return required_bits
