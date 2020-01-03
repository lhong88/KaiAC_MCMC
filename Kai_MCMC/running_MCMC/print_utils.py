# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:04:55 2018

Useful print information to track progress of fit (best walker, acceptance fraction, etc.)

And also a utility to ignore lsoda warnings that frequently come up when integrating the ODEs, 
taken from the following:
    https://stackoverflow.com/questions/31681946/disable-warnings-originating-from-scipy
"""
import time
import numpy as np
import os
import pandas as pd
import sys
import contextlib

###################################################
############# Save best walker information
###################################################



def make_best_walker_table(job_settings):
    '''
    Initializes a table that will display the best walker at every 
    printed-out step from the run_MCMC script.
    '''
    best_walker_table_csvfile, param_names = [job_settings[jp] for jp in \
             ['best_walker_table_csvfile', 'param_names']]
    df = pd.DataFrame(columns=np.concatenate(( ['regime'], param_names)))
    df.to_csv(best_walker_table_csvfile)
    return None


def print_save_best_walker_info(sampler, i, job_settings, tag):
    '''
    Given the sampler object, finds the best walker, prints info about it,
    saves it to the best_walker_table.
    i is the time index to use in the chain  to extract the best walker
    '''
    best_walker_table_csvfile, param_names = [job_settings[jp] for jp in \
             ['best_walker_table_csvfile', 'param_names']]
    # report best walker, log probability, acceptance fraction
    # i = -1 for _init chains
    # i = multiple of print_freq for _burnin and _iter chains
    best_walker = np.where(sampler.lnprobability[:,i] == np.max(sampler.lnprobability[:,i]))[0][0]
    print("Best walker is {}".format(sampler.chain[best_walker,i,:]))#[float('{:.3f}'.format(f)) for f in sampler.chain[best_walker,-1,:]]))
    print("ln prob is {:.3f}".format(sampler.lnprobability[best_walker, i]))
    print("acc frac is {:.3f}".format(np.mean(sampler.acceptance_fraction)))
        
    # save best walker
    # since I never use the csv best walker file I've commented out this section
    #walker_row = pd.Series( [tag] + list(sampler.chain[best_walker,i,:]), index=np.concatenate(( ['regime'], param_names)))
    #best_walker_table = pd.read_csv(best_walker_table_csvfile, index_col=0)
    #best_walker_table = best_walker_table.append(walker_row, ignore_index=True)
    #best_walker_table.to_csv(best_walker_table_csvfile)
    
    return best_walker
    
def print_elapsed_time(t0):
    m, s = divmod(time.clock()-t0, 60); h, m = divmod(m, 60)
    print("Time elapsed: {:02d}:{:02d}:{:02d}".format(int(h), int(m), int(s)))
    return None


###################################################
############# Ignore integration warnings
###################################################

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied: 
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied
