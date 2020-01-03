import pickle
import numpy as np

opts= np.asarray([pickle.load(open('output/optimized_walker_'+str(i)+'.p', 'r'))['x'] for i in range(10)])
opts_lnprob= np.asarray([pickle.load(open('output/optimized_walker_'+str(i)+'.p', 'r'))['fun'] for i in range(10)])

print opts_lnprob
np.save('output/optimized_walkers.npy', opts)
np.save('output/optimized_walkers_lnprob.npy', opts_lnprob)
