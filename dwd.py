import numpy as np
import os
from tqdm import tqdm

import ray
from utils import compute_snr
from raynest.parameter import LivePoint
from raynest.model import Model
from tdggs import Evolutioner

class DWD(Model):
    """
    """
    def __init__(self, time, data, signal_model, kwargs):
        self.time   = time
        self.data   = data
        self.signal = signal_model
        self.sigma  = kwargs['sigma_noise']
        self.names=['logA','logf','logfdot','phi']
#        self.bounds=[[-56,-50],[np.log(1./kwargs['T']),np.log(kwargs['srate']/2.)],[0.0,2*np.pi]]
        self.bounds=[[-56,-50],[-9.5,-5],[-50,-28],[0.0,2*np.pi]]

    def log_likelihood(self, population):
        """
        standard unit gaussian likelihood
        """
        if len(population) == 0:
            return -0.5*np.sum(self.data**2/self.sigma**2)
            
        s = np.array([self.signal(self.time,individual.values) for individual in population])
#        snrs = np.array([compute_snr(si,self.sigma) for si in s])
#        mask = snrs > 0.0
#        amplitudes = np.array([np.exp(individual['logA']) for individual in population])[~mask]
#        sigma_sq = np.sum(amplitudes**2)+self.sigma**2
#        exit()
        r = (self.data - np.sum(s,axis=0))#[mask]
        return -0.5*np.sum(r**2/self.sigma**2)-r.shape[0]*np.log(2*np.pi*self.sigma**2)/2
    
    def log_posterior(self, population):
        logP = 0.0
#        for p in population:
#            if not(np.isfinite(super(DWD,self).log_prior(p))):
#                return -np.inf

        return logP+self.log_likelihood(population)

if __name__ == "__main__":
    file = "/Users/wdp/Documents/projects_ongoing/genetic_search/galactic_data/DWD_pop_agCE_Pmax025d_KroupaIMF_BPsfh_inputLDC.hdf5"
    import h5py
    
    sampling_parameters_names = ['Amplitude',
                                 'Frequency',
                                 'FrequencyDerivative',
                                 'InitialPhase']
    
    injection_parameters = list()
    with h5py.File(file,'r') as f:
        for n in sampling_parameters_names:
            injection_parameters.append(f['H5LISA']['GWSources']['GalBinaries'][n][()])

    injection_parameters = np.array(injection_parameters)
    
    from waveforms import sinusoid, logfdot
    import matplotlib.pyplot as plt
    from ray.experimental import tqdm_ray
    
    ray.init()
    Nsig  = 10000
    sigma_noise = 1e-22
    n_chunk = 1
    ndays   = 7
    decimation_factor = 1
    zero_noise = 1
    chunk_duration = ndays*86500
    rng = np.random.default_rng(314)
    
    T   = n_chunk*chunk_duration # in secs
    srate = 0.2/decimation_factor
    Npts  = int(T*srate)
    logb_th = 3.0
    output_folder = 'quick_run'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder+'/signals', exist_ok=True)
#    snr_min = 5
#    min_amp = np.log(sigma_noise*snr_min/np.sqrt(4.0/chunk_duration))
#    print(min_amp)
    min_amp = np.log(np.sqrt(2*logb_th*sigma_noise**2))
    noise   = rng.normal(0,sigma_noise,size=Npts)*(1-zero_noise)
    time    = np.linspace(0,T,Npts)
    idx     = rng.choice(injection_parameters.shape[1], size=Nsig, replace=False)
    # 0: amplitude 1: frequency 2: fdot 3: phase

    injection_parameters = injection_parameters[:,idx].T
    injection_parameters[:,0] = np.log(injection_parameters[:,0])
    injection_parameters[:,1] = np.log(injection_parameters[:,1])
    injection_parameters[:,2] = np.log(injection_parameters[:,2])
#    
#    from scipy.stats import linregress
#    plt.plot(injection_parameters[:,1],injection_parameters[:,2],'o')
#    logf = np.linspace(-9.5,-5,100)
#    fit = linregress(injection_parameters[:,1],injection_parameters[:,2])
#    print(fit)
#    plt.plot(logf,logfdot(logf,-18,3.4))
#    plt.show()
#    exit()
#    
    
    injected_signals = [sinusoid(time, pi) for pi in injection_parameters]
    snrs = [compute_snr(inj, sigma_noise) for inj in injected_signals]
    fig = plt.figure()
    ax  = fig.add_subplot(111)
#        ax.plot(ti,noise,'r',alpha=0.5,label='noise')
    ax.hist(snrs,bins=32,density=True,facecolor='black',alpha=0.5)
    ax.set_xlabel('snr')
    plt.savefig(output_folder+'/injected_snr.pdf',bbox_inches='tight')
#    injected_signals = np.atleast_2d(injected_signals[np.argmax(snrs)])
    idx, = np.where(np.array(snrs) > 8)
#    injected_signals = [injected_signals[i] for i in idx]

    sig  = np.sum(injected_signals,axis=0)
    data = noise+sig
    # split the data in chunks and analyse them one by one
    d = np.array_split(data, n_chunk)
    t = np.array_split(time, n_chunk)
    s = np.array_split(sig, n_chunk)
    
    fig = plt.figure()
    ax  = fig.add_subplot(111)
#        ax.plot(ti,noise,'r',alpha=0.5,label='noise')
    ax.plot(time,data,'b',alpha=0.5,label ='signals')
#    ax.plot(ti,di,'k',alpha=0.5,label ='data')
#    ax.plot(ti,np.sum([sinusoid(ti, pi.values) for pi in recovered_values],axis=0),linestyle = 'dashed', color='purple',alpha=0.5,lw=1.5,label='recovered')
#    plt.legend()
    plt.xlabel('time(s)')
    plt.ylabel('strain')
    plt.savefig(output_folder+'/data_{0:d}.pdf'.format(11),bbox_inches='tight')
        
    exit()
    
    jobs = []
    
    for i,ti,di,si in zip(range(n_chunk),t,d,s):#tqdm_ray.tqdm(,desc='chunk'):
        M    = DWD(ti, di, sinusoid, {'sigma_noise':sigma_noise, 'T':chunk_duration,'srate':srate,'min_amp':min_amp})
#        print("bounds = ", M.bounds)
#        print(np.min(freqs), np.max(freqs))
#        print(np.min(amps), np.max(amps))
#        exit()
        E    = Evolutioner.remote(M, seed = 666+i, n=1, mode='sample', burnin=0.75, output = os.path.join(output_folder,'chunk_{0:d}'.format(i)), position=i)
        
        maxL = np.sum(-0.5*(di-si)**2/sigma_noise**2)-di.shape[0]*np.log(2*np.pi*sigma_noise**2)/2
        print(maxL,np.sum(-0.5*(di)**2/sigma_noise**2)-di.shape[0]*np.log(2*np.pi*sigma_noise**2)/2)
        
        jobs.append(E.evolve.remote(n_generations = 100000))
        maxL = np.sum(-0.5*(di-si)**2/sigma_noise**2)
        print(maxL,np.sum(-0.5*(di)**2/sigma_noise**2))
    
    results = ray.get(jobs)
    ray.shutdown()

    for i,ti,di,si in tqdm(zip(range(n_chunk),t,d,s),desc='chunk'):
        recovered_values = results[i]#[-1]
        v   = np.column_stack(([pi.values[0] for pi in recovered_values],[pi.values[1] for pi in recovered_values],[pi.values[2] for pi in recovered_values], [pi.values[3] for pi in recovered_values]))
        np.savetxt(output_folder+'/signals/recovered_signals_chunk_{0:d}.txt'.format(i),v)

        fig = plt.figure()
        ax  = fig.add_subplot(111)
    #        ax.plot(ti,noise,'r',alpha=0.5,label='noise')
        ax.plot(ti,si,'b',alpha=0.5,label ='signals')
        ax.plot(ti,di,'k',alpha=0.5,label ='data')
        ax.plot(ti,np.sum([sinusoid(ti, pi.values) for pi in recovered_values],axis=0),linestyle = 'dashed', color='purple',alpha=0.5,lw=1.5,label='recovered')
        plt.legend()
        plt.xlabel('time(s)')
        plt.ylabel('strain')
        plt.savefig(output_folder+'/data_{0:d}.pdf'.format(i),bbox_inches='tight')
        
        nbins   = int(np.sqrt(len(recovered_values)))
        nbins_t = int(np.sqrt(Nsig))
        
        fig = plt.figure()
        ax  = fig.add_subplot(411)
        ax.hist([pi['logf'] for pi in recovered_values], density = True, facecolor='purple', alpha = 0.5, bins = nbins)
        ax.hist(injection_parameters[:,1], density = True, alpha = 0.5, facecolor='black', bins = nbins_t)
        ax.set_xlabel('frequency(Hz)')
        ax.set_xlim(M.bounds[1][0],M.bounds[1][1])
        ax = fig.add_subplot(412)
        ax.hist([pi['phi'] for pi in recovered_values], facecolor='purple', density = True, bins = nbins)
        ax.hist(injection_parameters[:,3], density = True, alpha = 0.5, facecolor='black', bins = nbins_t)
        ax.set_xlabel('phase')
        ax.set_xlim(M.bounds[3][0],M.bounds[3][1])
        ax = fig.add_subplot(413)
        ax.hist([pi['logA'] for pi in recovered_values], facecolor='purple', density = True, bins = nbins)
        ax.hist(injection_parameters[:,0], density = True, alpha = 0.5, facecolor='black', bins = nbins_t)
        ax.set_xlabel('amplitude')
        ax.set_xlim(M.bounds[0][0],M.bounds[0][1])
        ax = fig.add_subplot(414)
        ax.hist([pi['logfdot'] for pi in recovered_values], facecolor='purple', density = True, bins = nbins)
        ax.hist(injection_parameters[:,2], density = True, alpha = 0.5, facecolor='black', bins = nbins_t)
        ax.set_xlabel('log fdot')
        ax.set_xlim(M.bounds[2][0],M.bounds[2][1])
        plt.subplots_adjust(hspace=0.6)
        plt.savefig(output_folder+'/distributions_{0:d}.pdf'.format(i),bbox_inches='tight')
        
        from postprocess import read_samples
        
        file = "/Users/wdp/Documents/projects_ongoing/genetic_search/TDGGS/quick_run/chunk_0/samples.h5"

        p = read_samples(file)
        N = len(p)
        print("N samples = ",N)
        
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.scatter(injection_parameters[:,1],injection_parameters[:,0],color='turquoise',alpha=0.5,label='injection')
        for k in tqdm(range(N)):
            ax.scatter(p[k][:,1], p[k][:,0],color='purple',s=1,marker='o',rasterized=True, alpha=0.5)
        ax.set_ylabel('log(amplitude)')
        ax.set_xlabel('log(frequency(Hz))')
        ax.set_xlim(M.bounds[1][0],M.bounds[1][1])
        ax.set_ylim(M.bounds[0][0],M.bounds[0][1])
        plt.legend()
        plt.grid(alpha=0.5,linestyle='dotted')
        plt.savefig(output_folder+'/scatter_a_f_{0:d}.pdf'.format(i),bbox_inches='tight')

        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.scatter(injection_parameters[:,1],injection_parameters[:,2],color='turquoise',alpha=0.5,label='injection')
        for k in tqdm(range(N)):
            ax.scatter(p[k][:,1], p[k][:,2],color='purple',s=1,marker='o',rasterized=True, alpha=0.5)
        ax.set_ylabel('log(fdot)')
        ax.set_xlabel('log(frequency(Hz))')
        ax.set_xlim(M.bounds[1][0],M.bounds[1][1])
        ax.set_ylim(M.bounds[2][0],M.bounds[2][1])
        plt.legend()
        plt.grid(alpha=0.5,linestyle='dotted')
        plt.savefig(output_folder+'/scatter_f_fdot_{0:d}.pdf'.format(i),bbox_inches='tight')

