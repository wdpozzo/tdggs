import numpy as np

def compute_snr(h,sigma):
    return np.sqrt(np.sum(h**2/sigma**2)/2.)
    
# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
    """
    take a string of bits and translate it into a real number
    """
    decoded = list()
    largest = 2**n_bits
    for i in range(len(bounds)):
        # extract the substring
        start, end = i * n_bits, (i * n_bits)+n_bits
        substring = bitstring[start:end]
        # convert bitstring to a string of chars
        chars = ''.join([str(s) for s in substring])
        # convert string to integer
        integer = int(chars, 2)
        # scale integer to desired range
        value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
        # store
        decoded.append(value)
    return decoded

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
