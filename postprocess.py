import numpy as np
import h5py
import matplotlib.pyplot as plt

def read_samples(file):
    with h5py.File(file,'r') as f:
        group   = f.get('Samples')
        samples = [group[i][:] for i in group.keys()]
    return samples

def normalise(p, dx):
    return p/np.sum(p*dx)

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import joypy
    import pandas as pd
    from tqdm import tqdm
    # read the true distribution
    file = "/Users/wdp/Documents/projects_ongoing/genetic_search/galactic_data/DWD_pop_agCE_Pmax025d_KroupaIMF_BPsfh_inputLDC.hdf5"
    
    sampling_parameters_names = ['Amplitude',
                                 'Frequency',
                                 'FrequencyDerivative',
                                 'InitialPhase']
    
    injection_parameters = list()
    with h5py.File(file,'r') as f:
        for n in sampling_parameters_names:
            injection_parameters.append(f['H5LISA']['GWSources']['GalBinaries'][n][()])
    
    injection_parameters = np.array(injection_parameters)
    
    file = "/Users/wdp/Documents/projects_ongoing/genetic_search/TDGGS/test_priors/chunk_0/samples.h5"
    Ntrue = 300
    rng = np.random.default_rng(134314)
    idx = rng.choice(range(injection_parameters.shape[1]), size=Ntrue, replace=False)

    # 0: amplitude 1: frequency 2: fdot 3: phase

    injection_parameters = injection_parameters[:,idx].T
    injection_parameters[:,0] = np.log(injection_parameters[:,0])
    injection_parameters[:,1] = np.log(injection_parameters[:,1])
    injection_parameters[:,2] = np.log(injection_parameters[:,2])
    
    p = read_samples(file)
    sizes = [len(pi) for pi in p]
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.hist(sizes, density=True, bins = int(np.sqrt(len(sizes))), facecolor='purple', alpha = 0.5)
    ax.axvline(Ntrue)
    cr_n = np.percentile(sizes,[5,50,95])
    print(cr_n)
    ids = [np.abs(sizes-c).argmin() for c in cr_n]
    print(ids)
    bounds = [[-56,-50],[-9.5,-5],[-50,-28],[0.0,2*np.pi]]#[[-56,-50],[np.log(1./(5*86500)),np.log(0.02/2.)],[0.0,2*np.pi]]
    for i,par in enumerate(["logA","logf","logfdot","phi"]):
        f = plt.figure()
        ax = f.add_subplot(111)
        realisations = list()
        for k in tqdm(range(len(p))):
            n, edges = np.histogram(p[k][:,i], bins = np.linspace(bounds[i][0],bounds[i][1],int(np.sqrt(Ntrue))))
            realisations.append(n)
        
        realisations = np.array(realisations)
        print(realisations.shape)
        regions = np.percentile(realisations, [5,50,95], axis = 0)
        x = 0.5*(edges[1:]+edges[:-1])
        dx = np.diff(x)[0]
        ax.fill_between(x,normalise(regions[0],dx),normalise(regions[2],dx),facecolor='purple',alpha=0.5)
        ax.plot(x, normalise(regions[1],dx), lw=0.7, color='k')
        ax.hist(injection_parameters[:,i], bins = np.linspace(bounds[i][0],bounds[i][1],int(np.sqrt(Ntrue))), facecolor='black', alpha=0.5, density=True)
        ax.set_xlabel(par)
        fig, axes = joypy.joyplot(pd.DataFrame(realisations[rng.choice(range(realisations.shape[0]),replace=False,size=10),:].T),
                          range_style='own', grid="y", linewidth=1, legend=False, fade=True,
                          figsize=(6,5))
    plt.show()
