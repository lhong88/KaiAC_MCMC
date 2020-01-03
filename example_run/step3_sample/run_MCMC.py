'''
Example MCMC code that fits a double site model to 
    autophosphorylation data.
Danylo L May 2018
Should be Python 2, 3 compatible
'''
#####################################################################
###### Import modules, utilities         
#####################################################################

import os
from scipy import  stats
import numpy as np
import math
from emcee.utils import MPIPool
from mpi4py import MPI 
import sys
import pandas as pd
import datetime
from shutil import copyfile

# import functions that perform MCMC steps
from Kai_MCMC.running_MCMC.MCMC_steps import populate_walkers_from_prior,\
    do_initialization, do_burnin, do_sampling
from Kai_MCMC.general.simulate_rxns_v3 import simulate_autophos, simulate_dephos

#####################################################################
###### Select model      
#####################################################################

from Kai_MCMC.models.ds_v5.ds_model_features import get_tableKaiC
from Kai_MCMC.models.ds_v5.ds_model_derivatives import mod

dim = mod.n_tot_params

#####################################################################
###### Set settings          
#####################################################################

######################## saving options

# define output directory
current_time_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
saveto_directory = 'MCMC_ds_{}/'.format(current_time_string)
#if not os.path.exists(saveto_directory):
#    os.makedirs(saveto_directory)
# define prefix to all outputted files
saveto_filename_prefix = saveto_directory+'MCMC_ds'
best_walker_table_suffix = '_best_walker_table.csv'

out_script_suffix = '_script_{}.py'.format(current_time_string)

# print the contents of this script to console?
# (if running on midway, then the contents of this script will 
#    appear at the top of your .out file)
print_file = 1

# how many timesteps to skip in saving to files
save_frequency= 100

######################## parallelization / number of walkers

# do parallelization?
# don't forget to set this to 1 if running on Midway!
do_pool = 1
num_nodes= 8

# properties of Midway -- don't change these
ncores_per_node_midway1 = 16
ncores_per_node_midway2 = 28

# number of walkers
# if running on Midway, efficient parallelization is only achieved
# if the number of walkers is a multiple of the number of cores
nwalkers = num_nodes*ncores_per_node_midway2


######################## MCMC duration settings

# how long to sample at each annealing temperature (typically ~10k)
ninit = 10000
# how long to sample to let walkers equilibrate after the annealing steps
# (should NOT be used for inference) (typically ~20k)
nburnin = 20000
# how long to sample to collect statistics (typically ~40k)
niter =  30000


######################## structural MCMC settings

# weights on the likelihood during initialization/annealing
# (higher weight -> higher importance of likelihood over prior)
# should end at 1.
annealing_betas = np.arange(0.3, 1.1, 0.1)

# magnitude of stretch move when proposing steps
stretchfactor = 1.1

# width of ball around which walkers drawn from at start of burnin
burnin_ball_width = 0.001

######################## minor printing options

# when drawing walkers from prior, how often to print progress
print_every_n_walkers_drawn = 10
# for burn-in and sampling, how often to print/save info
how_many_sampling_prints = 5 

#####################################################################
###### Import data           
#####################################################################

# autophosphorylation data
autophos_file = '../../data/autophos_DL_EL_2017.csv'
autophos_table = pd.read_csv(autophos_file)

# 30 degree, 5 mM total nucleotide autophosphorylation data codes
auto_rxns = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6',
        'r7', 'r8', 'r9', 'r10', 'r11', 'r12',
        'r13', 'r14', 'r15', 'r16', 'r17', 'r18']

# Which reactions to use to fit?
auto_rxns_to_fit = auto_rxns

# dephosphorylation data (Rust et al., Science, 2007)
dephospho_data = pd.read_csv('../../data/dephospho_rust_2007.csv')

# Which reactions to use to generate walkers from prior?
# (should be subset of auto_rxns_to_fit just to speed things up)
rxns_to_gen_walkers_from_prior = ['r1', 'r7', 'r8']

pforms_to_fit = ['U', 'T', 'D']

# weight on dephosphorylation data
# if zero, doesn't fit to dephospho at all
# data has 21 points compared to 8 in each autophos reaction, so 
# a weight of 8/21 makes dephospho data same weight as a single autophos reaction,
# a weight of 1 makes every available data point as equal weight
dephos_ll_scale = 4.

# hydrolysis data from Terauchi et al., PNAS, 2007
adp_per_day_KaiC= 29.8
adp_per_day_KaiC_std= 5.1
hydrolysis_ll_scale= 1.
# only impose penalty for auto-phos reactions where [KaiA] <= 1.5 uM
hydrolysis_penalty_set= ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9']

#####################################################################
###### Process settings, start things up       
##################################################################### 
    
# load settings defined above into a dictionary for easy access
# in functions found in other files
job_settings = {}
job_settings['dim'] = dim
job_settings['nwalkers'] = nwalkers
job_settings['param_names'] = mod.tot_params
job_settings['ninit'] = ninit
job_settings['niter'] = niter
job_settings['nburnin'] = nburnin
job_settings['annealing_betas'] = annealing_betas
job_settings['stretchfactor'] = stretchfactor
job_settings['auto_rxns_to_fit'] = auto_rxns_to_fit
job_settings['dephos_ll_scale'] = dephos_ll_scale
job_settings['rxns_to_gen_walkers_from_prior'] = rxns_to_gen_walkers_from_prior
job_settings['how_many_sampling_prints'] = how_many_sampling_prints
job_settings['print_every_n_walkers_drawn'] = print_every_n_walkers_drawn
job_settings['saveto_filename_prefix'] = saveto_filename_prefix
job_settings['best_walker_table_csvfile'] = saveto_filename_prefix + best_walker_table_suffix
job_settings['save_frequency']= save_frequency

# identify master node
if (MPI.COMM_WORLD.Get_rank()==0):
    master=True
else:
    master=False
    
# prints to console/.out file, if desired
#if print_file:
#    if (master):
        #copyfile(__file__, saveto_filename_prefix+out_script_suffix)
# print some basic model information before getting into the fit
if (master):
    print('\n\nfilename: {}'.format(saveto_filename_prefix))
    print('number of parameters: {}'.format(dim))

#####################################################################
###### MCMC Functions
#####################################################################
      
def logprior(q):
    '''
    The log prior contribution to the log posterior.
    Given a walker q, evaluates the probability of each parameter according
        to its prior, and returns the sum
    Fordidden values of parameters should result in -infinity!
    '''
    # The initial conditions have a flat prior, so we only need to check 
    # that they're non-negative
    return mod.calc_log_prior(q)

def loglike_autophos(rxn, q):
    '''
    A single autophos reaction's log likelihood contribution to the log posterior.
    Given a walker q and an autophos reaction rxn to simulate, integrates the ODEs
        to simulate the reaction, then returns the summed log probability of 
        (simulation - data) evaluated using N(0, sigma)
    '''
    # extract relevant portion of autophosphorylation data
    subdat = autophos_table[autophos_table.rxn == rxn]
    # integrate the model here
    sol= simulate_autophos(subdat, q, mod, do_pool)
    if sol == False:
        return -np.inf
    else:
        auto_t, Xs, int_time= sol
    
    # check integration is reasonable
    
    if np.any(Xs < 0):
        if np.allclose(Xs[Xs < 0], 0):
            Xs[Xs < 0] = 0.
        else:
            return -np.inf
    if not np.allclose(np.sum(Xs[:, :-2], 1), 3.5):
        return -np.inf
    
    # compute P(sim - data) ~ N(0, sigma) for each timepoint and KaiC phosphoform,
    # return sum of log probabilities
    diffs = subdat[pforms_to_fit]*mod.KaiC/100. - get_tableKaiC(Xs, pforms_to_fit)
    # find which index of q corresponds to the global error term sigma
    sigma = q[mod.param_to_i['sig']]    
    phos_log_like= np.sum(stats.norm(scale=math.sqrt(sigma)).logpdf(diffs))
    
    # compute the log likelihood for the hydrolysis penalty
    Pi_log_like= 0.0
    
    if rxn in hydrolysis_penalty_set:
        reaction_time= auto_t[-1]/3600. # convert from second to hour
        scaled_hydrolysis_rate= (reaction_time/24.)*adp_per_day_KaiC*mod.KaiC
        scaled_hydrolysis_std= (reaction_time/24.)*adp_per_day_KaiC_std*mod.KaiC
        # add together the [Pi] as well as subset of [KaiC] that is bound to ADP
        tot_ADP= Xs[-1, -1] + np.sum(Xs[-1, [2,3,6,7,8,9,12,13,10,11,14,15]])
        if tot_ADP > scaled_hydrolysis_rate:
            diff= tot_ADP - scaled_hydrolysis_rate
            Pi_log_like= -diff**2/(2*scaled_hydrolysis_std)**2
        
    return phos_log_like + hydrolysis_ll_scale*Pi_log_like


def loglike_dephos(q):   
    '''
    Dephosphorylation log likelihood. 
    Simulates Rust 2007 dephosphorylation data, returns summed log probability
        (phosphorylate for 20 hours, remove KaiA, dephosphorylate + compare to data)
    '''
    # integrate the model here    
    sol= simulate_dephos(dephospho_data, q, mod, do_pool)
    if sol == False:
        return -np.inf
    else:
        phos_t, Xs_auto, dephos_t, Xs_dephos, int_time= sol
    
    # check integration is reasonable
    if np.any(Xs_dephos < 0):
        #print(Xs_dephos[-1])
        if np.allclose(Xs_dephos[Xs_dephos < 0], 0):
            Xs_dephos[Xs_dephos < 0] = 0.
        else:
            return -np.inf
    # extract "new" KaiC conc after removing KaiA-bound KaiC
    newKaiCconc = np.sum(Xs_dephos[0, :-2])
    if not np.allclose(np.sum(Xs_dephos[:, :-2], 1), newKaiCconc):
        return -np.inf
    
    # compute P(sim - data) ~ N(0, sigma) for each timepoint and KaiC phosphoform,
    # return sum of log probabilities
    diffs = dephospho_data[pforms_to_fit]*newKaiCconc/100. - get_tableKaiC(Xs_dephos, pforms_to_fit)
    
    # find which index of q corresponds to the global error term sigma
    sigma = q[mod.param_to_i['sig']]    
    return np.sum(stats.norm(scale=math.sqrt(sigma)).logpdf(diffs))


def logpost(q, auto_rxns_to_fit, beta):
    '''
    Compute log posterior from log likelihood contributions for each rxn and 
        from log prior contribution, returning -infinity for anything forbidden
    '''
    # compute prior
    logpri = logprior(q)
    if np.isnan(logpri) or np.isneginf(logpri):
        return -np.inf
    # add contributions from each autophos reaction
    ll_autophos = 0.
    for rxn in auto_rxns_to_fit:
        res = beta*loglike_autophos(rxn, q)
        ll_autophos += res
    if np.isnan(ll_autophos) or np.isneginf(ll_autophos):
        return -np.inf
    # dephosphorylation contribution
    ll_dephos = 0.
    if dephos_ll_scale > 0.:
        ll_dephos = beta * dephos_ll_scale * loglike_dephos(q)
    if np.isnan(ll_dephos) or np.isneginf(ll_dephos):
        return -np.inf
    # log posterior = log likelihood + log prior
    ll = ll_autophos + logpri + ll_dephos
    if np.isnan(ll) or np.isneginf(ll):
        return -np.inf
    return ll
    

# only after defining all the functions should you do this!
if do_pool:
    # Create pool object
    pool = MPIPool()
    # Non-master cores wait for instructions and then exit
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    else:
        print("Cores: {}".format(pool.comm.Get_size()))
        # the master core generate an output folder
        os.makedirs(saveto_directory)
        copyfile(__file__, saveto_filename_prefix+out_script_suffix)
else:
    pool = None
   
job_settings['pool'] = pool


#####################################################################
###### MCMC: Burn-in
#####################################################################

print("\nRe-initializing at best walker from initialization...\n")

temp= 0.6

bw= np.load('../step2_powell/output/optimized_walkers.npy')
n_opts= bw.shape[0]
bw_lnprob= -np.load('../step2_powell/output/optimized_walkers_lnprob.npy')
bw_prob= np.exp(bw_lnprob - np.max(bw_lnprob))**temp
walker_partition= np.rint(nwalkers*bw_prob/np.sum(bw_prob))

print('walker partition: ' + str(walker_partition))

if not np.allclose(np.sum(walker_partition), nwalkers):
    diff= int(np.sum(walker_partition) - nwalkers)
    bw_pos= np.argmax(walker_partition)
    walker_partition[bw_pos]-= diff

assert np.allclose(np.sum(walker_partition), nwalkers)

burnin_pos0 = np.vstack(np.array([np.random.normal(bw[i], scale=burnin_ball_width, size=(int(walker_partition[i]),dim)) for i in range(n_opts)]))
np.save(saveto_filename_prefix+'burnin_pos0.npy', burnin_pos0)

burnin_lnprob0= np.array([logpost(walker, auto_rxns_to_fit, 1.0) for walker in burnin_pos0]).reshape(len(burnin_pos0), 1)
np.save(saveto_filename_prefix+'burnin_pos0_lnprob.npy', burnin_lnprob0)

burnin_finalpos = do_burnin(burnin_pos0, job_settings, logpost)

#####################################################################
###### MCMC: Sampling
#####################################################################

do_sampling(burnin_finalpos, job_settings, logpost)

if do_pool:
    pool.close()

############ ----------------- End of code! -----------------
