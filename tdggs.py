import numpy as np
import ray
import copy
import os, sys
from tqdm import tqdm
import warnings
from numpy import inf
import numpy as np
import ray
from raynest.parameter import LivePoint
from raynest.model import Model
import struct
import h5py

def map_to_bit_string(value, lower_bound, upper_bound, num_bits):
    """
    Maps a real value from the interval [lower_bound, upper_bound]
    to a byte string with a fixed number of bits.

    :param value: The real value to map.
    :param lower_bound: The lower bound of the interval.
    :param upper_bound: The upper bound of the interval.
    :param num_bits: The number of bits for the byte string.
    :return: A byte string representation of the mapped value.
    """
    # Check that the value is within the specified bounds
    if not (lower_bound <= value <= upper_bound):
        raise ValueError(f"Value {value} is out of bounds ({lower_bound}, {upper_bound})")

    # Calculate the range and scale the value to [0, 2^num_bits - 1]
    range_size = upper_bound - lower_bound
    scaled_value = int((value - lower_bound) / range_size * (2**num_bits - 1))

    # Convert to a byte string
    bit_string = format(scaled_value, f'0{num_bits}b')

    return bit_string

def map_bit_string_to_real(bit_string, lower_bound, upper_bound, num_bits):
    """
    Maps a bit string with a fixed number of bits back to a real interval [lower_bound, upper_bound].

    :param bit_string: The bit string to map.
    :param lower_bound: The lower bound of the interval.
    :param upper_bound: The upper bound of the interval.
    :return: The mapped real value.
    """
    # Convert the bit string to an integer
    integer_value = int(bit_string, 2)

    # Normalize the integer to the range [0, 2^num_bits - 1]
    range_size = 2**num_bits - 1

    # Scale the normalized value to the interval [lower_bound, upper_bound]
    scaled_value = lower_bound + (integer_value / range_size) * (upper_bound - lower_bound)

    return scaled_value

def bin2float(binary_string):

    # Convert binary string to an integer
    integer_representation = int(binary_string, 2)
    
    # Pack the integer into bytes, assuming it's a 64-bit float
    byte_data = struct.pack('>Q', integer_representation)  # '>I' for big-endian unsigned int
    
    # Unpack the bytes as a float
    double_number = struct.unpack('>d', byte_data)[0]  # '>f' for big-endian float
    
    return double_number

def float2bin(f):
    ''' Convert float to 64-bit binary string.

    Attributes:
        :f: Float number to transform.
    '''
    byte_data = struct.pack('>d', f)
    binary_string = ''.join(f'{byte:08b}' for byte in byte_data)
    
    return binary_string

def search_rule(a, b, rng, death_rate, birth_rate, mutation_rate, move):

    p_t = log_transition_probabilities(death_rate, birth_rate, mutation_rate, move)

    if a-b + p_t > 0:
        return True
    return False

def metropolis_hastings(a, b, rng, death_rate, birth_rate, mutation_rate, move):

    p_t = log_transition_probabilities(death_rate, birth_rate, mutation_rate, move)

    if (a-b + p_t) > np.log(rng.uniform(0,1)):
        return True
    return False


def log_transition_probabilities(death_rate, birth_rate, mutation_rate, move):

    total_rate = death_rate + birth_rate + mutation_rate
    
    if total_rate == 0:
        return 0.0
        
    P_birth = birth_rate / total_rate
    P_death = death_rate / total_rate
    P_no_change = 1 - P_birth - P_death

    # die,sex,mut,plus,drift
    
    if np.array_equal(move,[1,0,0,0,0]):
        return np.log(P_death)
    elif np.array_equal(move,[0,1,0,0,0]) or np.array_equal(move,[0,0,0,1,0]):
        return np.log(P_birth)
    else:
        return np.log(P_no_change)

@ray.remote(num_cpus=1)
class Evolutioner:
    def __init__(self, model, seed=None, n=1, mode = 'search', burnin = 0.5, output='/', position = 1, num_bits = 64):
        
        self.rng            = np.random.default_rng(seed)
        self.model          = model
        self.population     = [model.new_point(rng = self.rng) for _ in range(n)]
        # die,sex,mut,plus,drift
        self.alpha0         = np.array([2,1,1,1,1])
        self.alpha          = self.alpha0.copy()
        self.bounds         = self.model.bounds
        self.burnin_frac    = burnin
        self.num_bits       = num_bits
        self.mutation_rate  = 1.0 / (float(self.num_bits)* len(self.bounds))
        self.mutation_strength = 0.2
        self.mode           = mode
        self.position       = position
        self.samples        = []
        self.birth_rate     = (self.alpha0[1]+self.alpha0[3])
        self.death_rate     = self.alpha0[0]
        self.drifting_rate  = (self.alpha0[2]+self.alpha0[4])
        self.total_rate     = self.birth_rate+self.death_rate+self.drifting_rate
        self.setup_output(output)
        
    def setup_output(self, output):
        os.makedirs(output, exist_ok = True)
        self.file  = h5py.File(os.path.join(output,'samples.h5'), 'w')
        self.group = self.file.create_group('Samples')

    def evolve(self, n_generations = 10):
        
        acc = 0
        rej = 1
        
        if self.mode == 'search':
            acceptance_rule = search_rule
        elif self.mode == 'sample':
            acceptance_rule = metropolis_hastings
            
        logP0           = self.model.log_likelihood(self.population)
        self.burnin     = int(self.burnin_frac*n_generations)
        
        for gen in range(n_generations):#tqdm(range(n_generations), desc="generation ->", ascii=True, position = self.position):
            
            logP, trial, move   = self.evolve_one_step()

            #if gen < self.burnin:
            #    acceptance_rule = search_rule
            #else:
            #    acceptance_rule = metropolis_hastings

            if acceptance_rule(logP, logP0, self.rng, self.death_rate, self.birth_rate, self.drifting_rate, move):
                self.population = copy.deepcopy(trial)
                logP0           = logP
                acc            += 1
                self.alpha     += move
                print('generation {0:3d}/{1:4d} - acc {2:0.3f} - best population = {3:3d} - log_post = {4:.5f} - b = {5:.5f} - d = {6:.5f} - m = {7:.5f} moves = {8}'.format(gen+1,
                       n_generations,acc/float(acc+rej),len(self.population),logP0,
                       self.birth_rate/self.total_rate,self.death_rate/self.total_rate,self.drifting_rate/self.total_rate,self.alpha))
                self.N = len(self.population)
                
                if gen > self.burnin:
#                    self.samples.append(trial)
                    self.group.create_dataset('{0:d}'.format(gen), data = np.row_stack([t.values for t in trial]))
                #else:
                #    self.alpha += move
                #    self.birth_rate     = (self.alpha[1]+self.alpha[3])
                #    self.death_rate     = self.alpha[0]
                #    self.drifting_rate  = (self.alpha[2]+self.alpha[4])
                #    self.total_rate     = self.birth_rate+self.death_rate+self.drifting_rate
            else:
                rej += 1
           
        sys.stderr.write('\n')
        
        self.file.close()
        return self.population

    def evolve_one_step(self):
        """
        evolve a random individual from the population
        """
        population = copy.deepcopy(self.population)
        N = len(population)
        
        if N != 0:
            
            idx                    = self.rng.integers(0,N)
            p                      = population[idx]
            probs                  = self.rng.dirichlet(self.alpha0)
            die,sex,mut,plus,drift = self.rng.multinomial(1,probs)

            if sex == 1:
                child = self.crossover(p, population[self.rng.integers(0,len(population))])
                population.append(child)
                logP = self.model.log_posterior(population)
                child.logL = logP
                child.logP = 1
                return logP, population, np.array((die,sex,mut,plus,drift))
                
            elif mut == 1:
                n = self.mutation(p)
                population.pop(idx)
                population.append(n)

            elif die == 1:
                population.pop(idx)
                logP = self.model.log_posterior(population)
                return logP, population, np.array((die,sex,mut,plus,drift))

            elif plus == 1:
                n = self.model.new_point(rng = self.rng)
                population.append(n)
                
            elif drift == 1:
                n = self.drift(p)
                population.pop(idx)
                population.append(n)
#            else:
#                pass
        else:
            n = self.model.new_point(rng = self.rng)
            population.append(n)
            die,sex,mut,plus,drift = (0, 0, 0, 1, 0)
        
        logP = self.model.log_posterior(population)
        n.logL = logP
        n.logP += 1
        
        return logP, population, np.array((die,sex,mut,plus,drift))
        
    def mutation(self, x):

        y = x.copy()
        
        for i in range(len(y.values)):

            v = list(map_to_bit_string(x.values[i], self.bounds[i][0], self.bounds[i][1], self.num_bits)) #list(float2bin(y.values[i]))
            g = self.rng.choice([True, False], len(v), p=[self.mutation_rate,1-self.mutation_rate])
            
            for j in range(len(v)):
                if g[j]:
                    v[j] = str(1-int(v[j]))

            y.values[i] = map_bit_string_to_real(''.join(v), self.bounds[i][0], self.bounds[i][1], self.num_bits)

        return y

    def crossover(self, x, y):
        """
        cross the genome of two parents to produce a child
        """
        z = self.model.new_point(rng = self.rng)
        for i in range(len(x.values)):

            v1   = list(map_to_bit_string(x.values[i], self.bounds[i][0], self.bounds[i][1], self.num_bits))
            v2   = list(map_to_bit_string(y.values[i], self.bounds[i][0], self.bounds[i][1], self.num_bits))
            idx  = self.rng.choice([True, False], len(v1))
            f    = ''.join(np.where(idx, v1, v2))
            z.values[i] = map_bit_string_to_real(''.join(f), self.bounds[i][0], self.bounds[i][1], self.num_bits)
        
        return z

    def drift(self, x):
        """
        gaussian drift the genome
        """
        y = x.copy()
        D = len(y.values)
        for i in range(D):
            if self.rng.uniform() < self.mutation_rate:
                #dx = 1e-3*np.abs(self.bounds[i][0]-self.bounds[i][1])
                g  = self.rng.normal(0.0, self.mutation_strength)
                y.values[i] *= (1+g) 

        return y
