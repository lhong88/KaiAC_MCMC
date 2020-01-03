# -*- coding: utf-8 -*-
"""
Main functions for running MCMC:
    1. populate walkers from prior, filtering those that give reasonable integrations
    2. do initialization/annealing, incrementally boosting weight on likelihood
    3. do burnin (allow walkers to spread out and decorrelate)
    4. do sampling (generate statistics)
May 2018
"""
import emcee
import time
import numpy as np
import random

# useful functions for printing out progress of fit
from print_utils import make_best_walker_table, \
        print_save_best_walker_info, print_elapsed_time


def isbad(logpost, q, rxns_to_gen_walkers_from_prior):
    '''
    Helper function for populate_walkers_from_prior, determines if 
    a given walker gives a bad integration. Walkers are repeatedly drawn
    until they're no longer "bad" (i.e. until they return non -neg.inf likelihood)
    The reactions used for evaluation can just be a small subset of data to
    speed up computation.
    '''
    return np.isneginf( logpost(q, rxns_to_gen_walkers_from_prior, beta=1) )


def populate_walkers_from_prior(mod, job_settings, logpost):
    '''
    Outputs an array pos0 (n_walkers, dim) of walkers drawn from the parameter priors
    that result in reasonable integrations
    '''
    # import job settings
    nwalkers, dim, print_every_n_walkers_drawn, rxns_to_gen_walkers_from_prior = [job_settings[jp] for jp in \
             ['nwalkers', 'dim', 'print_every_n_walkers_drawn', 'rxns_to_gen_walkers_from_prior'] ]
    
    print("\nPopulating {:.0f} walkers from prior...\n".format(nwalkers))
    t0 = time.time()

    pos0 = np.zeros(shape=(nwalkers,dim))
    ngood = 0
    
    while ngood < nwalkers:
        #print("populate walkers_from_prior: Attempt!")
        # keep drawing a walker until its logposterior doesn't return -np.inf 
        draw_q = mod.generate_walker()
        if not isbad(logpost, draw_q, rxns_to_gen_walkers_from_prior):
            # add it to pos0 if integration reasonable
            pos0[ngood, :] = draw_q
            ngood += 1
            #print("populate_walkers_from_prior: good walker!")
            if ngood % print_every_n_walkers_drawn == 0:
                print("{} of {} drawn ({:.2f} sec)".format(ngood, nwalkers, time.time()-t0))
        #else:
        #    print("populate_walkers_from_prior: bad walker!")
    print("\n...{:.0f} walkers populated in {:.2f} sec\n".format(nwalkers, time.time() - t0))
    
    return pos0


def do_initialization(pos0, job_settings, logpost):
    '''
    Initialization/annealing scheme.
    Runs MCMC using logpost(beta) = beta*loglikelihood + logprior
    for a discrete, increasing set of betas (`annealing_betas` in job_settings)
    Returns last position of the chain and the best_walker_table for continual updating.
    
    I've made it so that the last initialization phase (beta = 1) is twice as long.
    '''
    
    # load job parameters
    ninit, how_many_sampling_prints, nwalkers, dim,\
        stretchfactor, auto_rxns_to_fit, pool, saveto_filename_prefix,\
            annealing_betas, param_names, save_frequency = [job_settings[jp] for jp in \
             ['ninit', 'how_many_sampling_prints', 'nwalkers', 'dim',\
              'stretchfactor', 'auto_rxns_to_fit', 'pool', 'saveto_filename_prefix',
              'annealing_betas', 'param_names', 'save_frequency'] ]
    
    
    # Initialize from the prior and do annealing
    t0 = time.clock()
    init_lnprobs = np.empty((nwalkers, 0))
    
    
    make_best_walker_table(job_settings)

    for i, beta in enumerate(annealing_betas):
        # run MCMC
        sampler = emcee.EnsembleSampler(nwalkers, dim, logpost, a=stretchfactor, args=(auto_rxns_to_fit, beta), pool=pool)
        # make the last initialization step longer
        if (beta - 1.0) <= 10**-4:
            sampler.run_mcmc(pos0, ninit*2)
        else:
            sampler.run_mcmc(pos0, ninit)
        
        # print progress and save chain of walkers
        print("\nStage {} of {}, beta = {:.2f}".format(i+1, 8, beta))
        print_elapsed_time(t0)
        
        # store log prob for each walker
        np.save(saveto_filename_prefix+'_CHAIN_init_%d' % i, sampler.chain[:, (save_frequency-1)::save_frequency, :])
        init_lnprobs = np.concatenate((init_lnprobs, sampler.lnprobability[:, (save_frequency-1)::save_frequency]), axis=1)
        np.save(saveto_filename_prefix+'_LNPROB_init', init_lnprobs)
        
        best_walker = print_save_best_walker_info(sampler, -1,\
                                                  job_settings, tag='init'+str(beta))
        
        ### important!!!
        # set pos0 to start the next run
        pos0 = sampler.chain[:,-1,:] # just continue where left off if still in initialization
        if i == (len(annealing_betas) - 1):
            pos0 = sampler.chain[best_walker,-1,:] # take position of best walker
            
    return pos0
        

def do_burnin(pos0, job_settings, logpost):
    '''
    Burnin steps.
    Returns last position of the chain and the best_walker_table for continual updating.
    '''
        
    # load job parameters
    nburnin, how_many_sampling_prints, nwalkers, dim,\
        stretchfactor, auto_rxns_to_fit, pool, saveto_filename_prefix, save_frequency = [job_settings[jp] for jp in \
             ['nburnin', 'how_many_sampling_prints', 'nwalkers', 'dim',\
              'stretchfactor', 'auto_rxns_to_fit', 'pool', 'saveto_filename_prefix', 'save_frequency'] ]
    
    print("\nDoing {} steps of burn-in...\n".format(nburnin))

    printfreq = max(1, nburnin/how_many_sampling_prints)
    
    beta = 1
    
    sampler = emcee.EnsembleSampler(nwalkers, dim, logpost, a=stretchfactor, args=(auto_rxns_to_fit, beta), pool=pool)
    t0 = time.clock()
    for i, result in enumerate(sampler.sample(pos0, iterations=nburnin)):       # what is sampler.sample?
        # periodically print, save information
        if i % printfreq == 0:
            print("\n{0:5.1%}".format(float(i) / nburnin))
            print_elapsed_time(t0)
            # save chain, lnprob to file
            np.save(saveto_filename_prefix+'_CHAIN_burnin', sampler.chain[:, (save_frequency-1)::save_frequency, :])
            np.save(saveto_filename_prefix+'_LNPROB_burnin', sampler.lnprobability[:, (save_frequency-1)::save_frequency])
            # print info on best walker, lnprob
            best_walker = print_save_best_walker_info(sampler, i,\
                                                      job_settings, tag='burnin'+str(i))
            
    print("\nBurn-in complete\n")
    
    best_walker = np.where(sampler.lnprobability[:,-1] == np.max(sampler.lnprobability[:,-1]))[0][0]
    print("Best walker is %s" % sampler.chain[best_walker,-1,:])
    print("max ln prob: %f" % max(sampler.lnprobability[:, -1]))
    
    
    
    np.save(saveto_filename_prefix+'_CHAIN_burnin', sampler.chain[:, (save_frequency-1)::save_frequency, :])
    np.save(saveto_filename_prefix+'_LNPROB_burnin', sampler.lnprobability[:, (save_frequency-1)::save_frequency])
    
    last_pos = sampler.chain[:, -1, :] 
    
    return last_pos


def do_sampling(pos0, job_settings, logpost):
    '''
    Sampling (or, I call it 'iteration' sometimes)
    Returns last position of the chain and the best_walker_table for continual updating.
    '''
        
    # load job parameters
    niter, how_many_sampling_prints, nwalkers, dim,\
        stretchfactor, auto_rxns_to_fit, pool, saveto_filename_prefix, save_frequency = [job_settings[jp] for jp in \
             ['niter', 'how_many_sampling_prints', 'nwalkers', 'dim',\
              'stretchfactor', 'auto_rxns_to_fit', 'pool', 'saveto_filename_prefix', 'save_frequency'] ]
    
    print("\nSampling from posterior for {} steps...\n".format(niter))
        
    printfreq = max(1, niter/how_many_sampling_prints)
    
    beta = 1
    
    sampler = emcee.EnsembleSampler(nwalkers, dim, logpost, a=stretchfactor, args=(auto_rxns_to_fit, beta), pool=pool)
    t0 = time.clock()
    for i, result in enumerate(sampler.sample(pos0, iterations=niter)):
        # periodically print, save information
        if i % printfreq == 0:
            print("\n{0:5.1%}".format(float(i) / niter))
            print_elapsed_time(t0)
            # save chain, lnprob to file
            np.save(saveto_filename_prefix+'_CHAIN_iter', sampler.chain[:, (save_frequency-1)::save_frequency, :])
            np.save(saveto_filename_prefix+'_LNPROB_iter', sampler.lnprobability[:, (save_frequency-1)::save_frequency])
            # print info on best walker, lnprob
            best_walker = print_save_best_walker_info(sampler, i,\
                                                      job_settings, tag='iter'+str(i))
    
    print("\nSampling complete\n")
    
    best_walker = np.where(sampler.lnprobability[:,-1] == np.max(sampler.lnprobability[:,-1]))[0][0]
    print("Best walker is %s" % sampler.chain[best_walker,-1,:])
    print("max ln prob: %f" % max(sampler.lnprobability[:, -1]))
    
    # Save the sample
    np.save(saveto_filename_prefix+'_CHAIN_iter', sampler.chain[:, (save_frequency-1)::save_frequency, :])
    np.save(saveto_filename_prefix+'_LNPROB_iter', sampler.lnprobability[:, (save_frequency-1)::save_frequency])
    np.save(saveto_filename_prefix+'_ACCFRAC', sampler.acceptance_fraction)
    
    last_pos = sampler.chain[:, -1, :] 
    
    return last_pos

