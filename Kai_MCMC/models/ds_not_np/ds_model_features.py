# -*- coding: utf-8 -*-
"""
Here's the central vault of common features of all double phosphosite models.
This file defines:
    state variables, with simple string names and LaTeX formatted names
        (to show up nicely in matplotlib)
    indices of state variables used for estimating initial conditions
    total concentration of KaiC used in simulations
    given an integrated solution to some double phosphosite ODEs,
        functions that extract phosphoforms of KaiC by summing up 
        the right columns 
May 2018
"""
import numpy as np

#####################################################################
############ STATE VARIABLES
#####################################################################

# the identity of the state variables. There's 16 KaiC variables 
# (4 phosphoforms, each can have ATP or ADP bound, and KaiA bound or not), so
# 4*2*2 = 16, and then free KaiA is the 17th.

state_vars = ['CTu', 'CDu', 'CTt', 'CDt',
             'ACTu', 'ACDu', 'ACTt', 'ACDt',
             'CTs', 'CDs', 'CTd', 'CDd',
             'ACTs', 'ACDs', 'ACTd', 'ACDd',
             'A', 'Pi']


formatted_state_vars_short_nuc = [r'$C_T^U$', r'$C_D^U$', r'$C_T^T$', r'$C_D^T$',
                      r'$^AC_T^U$', r'$^AC_D^U$', r'$^AC_T^T$', r'$^AC_D^T$',\
                      r'$C_T^S$', r'$C_D^S$', r'$C_T^D$', r'$C_D^D$',
                      r'$^AC_T^S$', r'$^AC_D^S$', r'$^AC_T^D$', r'$^AC_D^D$',
                      r'$A$', r'$P_i$']


formatted_state_vars = [r'$C_{ATP}^U$', r'$C_{ADP}^U$', r'$C_{ATP}^T$', r'$C_{ADP}^T$',
                      r'$^AC_{ATP}^U$', r'$^AC_{ADP}^U$', r'$^AC_{ATP}^T$', r'$^AC_{ADP}^T$',
                      r'$C_{ATP}^S$', r'$C_{ADP}^S$', r'$C_{ATP}^D$', r'$C_{ADP}^D$',
                      r'$^AC_{ATP}^S$', r'$^AC_{ADP}^S$', r'$^AC_{ATP}^D$', r'$^AC_{ADP}^D$',
                      r'$A$', r'$P_i$']


#####################################################################
############ INITIAL CONDITIONS
#####################################################################

# if a fit estimates initial conditions, these are the indices of the KaiC 
# states that are estimated at t = 0. Basically, it's all the non-KaiA bound
# states except for the last one because we know all the KaiC states must 
# sum to 3.5 or other total
init_indices = [0,1,2,3,8,9,10,11]

#####################################################################
############ KAIC PHOSPHOFORM EXTRACTION
#####################################################################

def sum_cols(X, cols):
    '''
    Given an array X, this is a simple utility that sums up columns
    according to indices passed in through cols. Useful for summing
    up all states of a given phosphoform, for example.
    X can either be a numpy vector or array
    '''
    is_X_vec = len(X.shape) == 1
    # add first desired column
    if is_X_vec:
        res = np.copy(X[cols[0]])
    else:
        res = np.copy(X[:, cols[0]])
    # keep adding more desired columns
    for c in cols[1:]:
        if is_X_vec:
            res += X[c]
        else:
            res += X[:, c]
    return res

#### The indices used in the following functions must correspond
#### to the order of states specified in state_vars and the model diagram!

def get_UKaiC(X):
    return sum_cols(X, [0,1,4,5])
def get_TKaiC(X): 
    return sum_cols(X, [2,3,6,7])
def get_SKaiC(X): 
    return sum_cols(X, [8,9,12,13])
def get_DKaiC(X):
    return sum_cols(X, [10,11,14,15])
def get_AKaiC(X):
    return sum_cols(X, [4,5,6,7,12,13,14,15])


def fun_from_pform(pform):
    '''
    Simple switch  utility for get_tableKaiC to map
    a phosphoform to a function that extracts the right indices
    to sum up (all defined above).
    '''
    fun = get_UKaiC
    if pform == 'T':
        fun = get_TKaiC
    elif pform == 'S':
        fun = get_SKaiC
    elif pform == 'D':
        fun = get_DKaiC
    return fun

def get_tableKaiC(X, pforms):
    '''
    Given an integrated solution array X, with dimensions
    n_t * n_state_var (where n_t: # timepoints), this function 
    outputs an n_t * n_pforms array to match with data.
    For instance, if you'd like to fit to U, T, and D KaiC, after 
    integrating the ODEs (X), you'd call 
      get_tableKaiC(X, ['U', 'T', 'D']) 
    and compare its elements to experimental U-, T-, D-KaiC
    '''
    # simply map phosphoform string to function that extracts
    # the right column indices of X to sum up.
    tab = fun_from_pform(pforms[0])(X)
    for i in range(len(pforms)-1):
        tab = np.vstack((tab, fun_from_pform(pforms[i+1])(X)))
    return tab.T
