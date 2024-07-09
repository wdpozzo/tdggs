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
from utils import decode
import struct
import h5py

def bin2float(b):
    ''' Convert binary string to a float.

    Attributes:
        :b: Binary string to transform.
    '''
    h = int(b,2).to_bytes(8, byteorder="big")
    return struct.unpack('>d', h)[0]

def float2bin(f):
    ''' Convert float to 64-bit binary string.

    Attributes:
        :f: Float number to transform.
    '''
    [d] = struct.unpack(">Q", struct.pack(">d", f))
    return f'{d:064b}'

def search_rule(a,b,*args,**kwargs):
    if a > b:
        return True
    return False

def metropolis_hastings(a,b,rng,*args,**kwargs):
    if a - b > np.log(rng.uniform(0,1)):
        return True
    return False

@ray.remote(num_cpus=1)
class Evolutioner:
    def __init__(self, model, seed=None, n=1, mode = 'search', burnin = 0.5, output='/', position = 1):
        
        self.rng            = np.random.default_rng(seed)
        self.model          = model
        self.population     = [model.new_point(rng = self.rng) for _ in range(n)]
        self.alpha0         = np.array([2,1,1,1,1])
        self.alpha          = self.alpha0.copy()
        self.bounds         = self.model.bounds
        self.burnin_frac    = burnin
        self.n_bits         = 64
        self.mutation_rate  = 1.0 / (float(self.n_bits) * len(self.bounds))#
        self.mode           = mode
        self.position       = position
        self.samples        = []
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
            
            logP, trial, move = self.evolve_one_step()

            if acceptance_rule(logP, logP0, self.rng):
                self.population = copy.deepcopy(trial)
                logP0           = logP
                self.alpha     += move
                acc            += 1
                if gen > self.burnin:
#                    self.samples.append(trial)
                    self.group.create_dataset('{0:d}'.format(gen), data = np.row_stack([t.values for t in trial]))
            
            else:
                rej += 1
            
            birth_rate = self.alpha[0]/gen
            death_rate = self.alpha[1]/gen
            print('generation {0:3d}/{1:4d} - acc {2:0.3f} - best population = {3:3d} - log_post = {4:.5f} - birth rate = {5:.5f} - death rate = {6:.5f}'.format(gen+1,n_generations,acc/float(acc+rej),len(self.population),logP0,birth_rate,death_rate))
            self.N = len(self.population)
        sys.stderr.write('\n')
        

        self.file.close()
        return self.population

    def evolve_one_step(self):
        """
        evolve a random individual from the population
        TODO:
            death probability should take into account low amplitudes
            local wander step
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
                logP = self.model.log_likelihood(population)
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
            v = list(float2bin(y.values[i]))
            g = self.rng.choice([True, False], len(v), p=[self.mutation_rate,1-self.mutation_rate])

            for j in range(len(v)):
                if g[j]:
                    v[j] = str(1-int(v[j]))

            y.values[i] = bin2float(''.join(v))
            
        return y

    def crossover(self, x, y):
        """
        cross the genome of two parents to produce a child
        """
        z = self.model.new_point(rng = self.rng)
        for i in range(len(x.values)):

            v1   = list(float2bin(x.values[i]))
            v2   = list(float2bin(y.values[i]))
            idx  = self.rng.choice([True, False], len(v1))
            f    = ''.join(np.where(idx, v1, v2))
            z.values[i] = bin2float(f)
        
        return z

    def drift(self, x):
        """
        gaussian drift the genome
        """
        y = x.copy()
        
        for i in range(len(y.values)):
            dx = np.abs(self.bounds[i][0]-self.bounds[i][1])/100.
            g = self.rng.normal(0.0, self.mutation_rate*dx)
            y.values[i] += g

        return y
