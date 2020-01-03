# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from scipy import stats, integrate
import odespy
import sys
if __name__ == '__main__':
    sys.path.append('../')
    sys.path.append('../../running_MCMC')
    sys.path.append('./')
    from ds_model_features import state_vars, formatted_state_vars, init_indices
    from model_class import Model_Pi
    from print_utils import stdout_redirected
else:
    from .ds_model_features import state_vars, formatted_state_vars, init_indices
    from ..model_class import Model_Pi
from numba import jit
from time import time


########################################
###### Model parameter names
########################################

param_names = [
    'k_r_D', 'k_h', 'k_a', 'k_b', 'k_p', 'k_d', #basic
    'dk_h_T', 'dk_h_S', 'dk_h_D', #phos on hydrolysis
    'dk_T_AT', 'dk_T_AS', 'dk_T_AD', #KaiA+phos on nuc. exch.
    'dk_h_AU', 'dk_h_AT', 'dk_h_AS', 'dk_h_AD', # KaiA+phos on hydrolysis
    'dk_a_UDP', 'dk_b_UDP', 'dk_b_TDP', 'dk_b_SDP', 'dk_a_DDP', 'dk_b_DDP', 'dk_b_TTP', 'dk_b_STP', 'dk_a_DTP', 'dk_b_DTP', #nuc+phos on KaiA on/off
    'dp_p_US', 'dk_d_SU', 'dk_p_TD', 'dk_d_DT', 'dk_p_SD', 'dk_d_DS', #phos on phos/dephos
    'dk_p_AUT', 'dk_d_ATU', 'dk_p_ATD', 'dk_d_ADT', 'dk_p_ASD', 'dk_d_ADS', 'dk_p_AUS', 'dk_d_ASU', #KaiA/phos on phos/dephos
    'sig' #global error
    ]

formatted_param_names = param_names

########################################
###### Model equations
########################################

@jit(cache=True)
def converter(q, ATPfrac, Temp):
    '''
    convert the raw parameters into rate constants
    '''
    
    # pass in all parameters and exponentiate 
    # make sure order of parameters here matches the order in _param_names!
    k_r_D, k_h, k_a, k_b, k_p, k_d, \
    dk_h_T, dk_h_S, dk_h_D, \
    dk_T_AT, dk_T_AS, dk_T_AD, \
    dk_h_AU, dk_h_AT, dk_h_AS, dk_h_AD, \
    dk_a_UDP, dk_b_UDP, dk_b_TDP, dk_b_SDP, dk_a_DDP, dk_b_DDP, dk_b_TTP, dk_b_STP, dk_a_DTP, dk_b_DTP, \
    dk_p_US, dk_d_SU, dk_p_TD, dk_d_DT, dk_p_SD, dk_d_DS, \
    dk_p_AUT, dk_d_ATU, dk_p_ATD, dk_d_ADT, dk_p_ASD, dk_d_ADS, dk_p_AUS, dk_d_ASU = 10**q[:40]
    
    # set K_on to 1 so that the on rates for ATP and ADP are the same.
    K_on= 1
    
    # it is assumed that the total amount of nucleotide is 5 mM
    # here converted to uM to be consistent with protein concentration
    nuc_tot= 5000.
    ATP= ATPfrac*nuc_tot
    ADP= nuc_tot - ATP
    ATP_weighed_frac= ATP/(ATP + K_on*ADP)
    
    # calculate KaiA-mediated nucleotide exchange rates
    k_T_A = k_r_D*ATP_weighed_frac
    
    # detailed balance    
    dk_a_STP= (dk_b_STP*dk_a_DDP*dk_d_ADS)/(dk_b_DDP*dk_p_ASD)
    dk_a_TTP= (dk_b_TTP*dk_a_DDP*dk_d_ADT)/(dk_b_DDP*dk_p_ATD)
    dk_a_TDP= (dk_b_TDP*dk_p_AUT)/dk_d_ATU
    dk_a_SDP= (dk_b_SDP*dk_p_AUS)/dk_d_ASU
    
    return np.array([k_T_A, k_h, k_a, k_b, k_p, k_d, \
                     dk_h_T, dk_h_S, dk_h_D, \
                     dk_T_AT, dk_T_AS, dk_T_AD, \
                     dk_h_AU, dk_h_AT, dk_h_AS, dk_h_AD, \
                     dk_p_US, dk_d_SU, dk_p_TD, dk_d_DT, dk_p_SD, dk_d_DS, \
                     dk_a_UDP, dk_a_TDP, dk_a_SDP, dk_a_DDP, dk_a_TTP, dk_a_STP, dk_a_DTP, \
                     dk_b_UDP, dk_b_TDP, dk_b_SDP, dk_b_DDP, dk_b_TTP, dk_b_STP, dk_b_DTP, \
                     dk_p_AUT, dk_d_ATU, dk_p_ATD, dk_d_ADT, dk_p_ASD, dk_d_ADS, dk_p_AUS, dk_d_ASU])


@jit(cache=True)
def ddt(X, t, converted_q):
    
    k_T_A, k_h, k_a, k_b, k_p, k_d, \
    dk_h_T, dk_h_S, dk_h_D, \
    dk_T_AT, dk_T_AS, dk_T_AD, \
    dk_h_AU, dk_h_AT, dk_h_AS, dk_h_AD, \
    dk_p_US, dk_d_SU, dk_p_TD, dk_d_DT, dk_p_SD, dk_d_DS, \
    dk_a_UDP, dk_a_TDP, dk_a_SDP, dk_a_DDP, dk_a_TTP, dk_a_STP, dk_a_DTP, \
    dk_b_UDP, dk_b_TDP, dk_b_SDP, dk_b_DDP, dk_b_TTP, dk_b_STP, dk_b_DTP, \
    dk_p_AUT, dk_d_ATU, dk_p_ATD, dk_d_ADT, dk_p_ASD, dk_d_ADS, dk_p_AUS, dk_d_ASU= converted_q
    
    
    X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, A= X[:17]
    
    rate= np.array([-(k_h + k_a*A + k_p*dk_p_US + k_p)*X0 + k_d*X3 + k_b*X4 + k_d*dk_d_SU*X9, \
                    k_h*X0 - k_a*dk_a_UDP*A*X1 + k_b*dk_b_UDP*X5, \
                    -(k_h*dk_h_T + k_a*dk_a_TTP*A + k_p*dk_p_TD)*X2 + k_b*dk_b_TTP*X6 + k_d*dk_d_DT*X11, \
                    k_p*X0 + k_h*dk_h_T*X2 - (k_d + k_a*dk_a_TDP*A)*X3 + k_b*dk_b_TDP*X7, \
                    k_a*A*X0 - (k_b + k_h*dk_h_AU + k_p*dk_p_US*dk_p_AUS + k_p*dk_p_AUT)*X4 + k_T_A*X5 + k_d*dk_d_ATU*X7 + k_d*dk_d_SU*dk_d_ASU*X13, \
                    k_a*dk_a_UDP*A*X1 + k_h*dk_h_AU*X4 - (k_b*dk_b_UDP + k_T_A)*X5, \
                    k_a*dk_a_TTP*A*X2 - (k_b*dk_b_TTP + k_h*dk_h_T*dk_h_AT + k_p*dk_p_TD*dk_p_ATD)*X6 + k_T_A*dk_T_AT*X7 + k_d*dk_d_DT*dk_d_ADT*X15, \
                    k_a*dk_a_TDP*A*X3 + k_p*dk_p_AUT*X4 + k_h*dk_h_T*dk_h_AT*X6 - (k_b*dk_b_TDP + k_d*dk_d_ATU + k_T_A*dk_T_AT)*X7, \
                    -(k_h*dk_h_S + k_a*dk_a_STP*A + k_p*dk_p_SD)*X8 + k_d*dk_d_DS*X11 + k_b*dk_b_STP*X12, \
                    k_p*dk_p_US*X0 + k_h*dk_h_S*X8 - (k_d*dk_d_SU + k_a*dk_a_SDP*A)*X9 + k_b*dk_b_SDP*X13, \
                    -(k_h*dk_h_D + k_a*dk_a_DTP*A)*X10 + k_b*dk_b_DTP*X14, \
                    k_p*dk_p_TD*X2 + k_p*dk_p_SD*X8 + k_h*dk_h_D*X10 - (k_d*dk_d_DT + k_d*dk_d_DS + k_a*dk_a_DDP*A)*X11 + k_b*dk_b_DDP*X15, \
                    k_a*dk_a_STP*A*X8 - (k_b*dk_b_STP + k_h*dk_h_S*dk_h_AS + k_p*dk_p_SD*dk_p_ASD)*X12 + k_T_A*dk_T_AS*X13 + k_d*dk_d_DS*dk_d_ADS*X15, \
                    k_p*dk_p_US*dk_p_AUS*X4 + k_a*dk_a_SDP*A*X9 + k_h*dk_h_S*dk_h_AS*X12 - (k_b*dk_b_SDP + k_d*dk_d_SU*dk_d_ASU + k_T_A*dk_T_AS)*X13, \
                    k_a*dk_a_DTP*A*X10 - (k_b*dk_b_DTP + k_h*dk_h_D*dk_h_AD)*X14 + k_T_A*dk_T_AD*X15, \
                    k_p*dk_p_TD*dk_p_ATD*X6 + k_a*dk_a_DDP*A*X11 + k_p*dk_p_SD*dk_p_ASD*X12 + k_h*dk_h_D*dk_h_AD*X14 - (k_b*dk_b_DDP + k_d*dk_d_DT*dk_d_ADT + k_d*dk_d_DS*dk_d_ADS + k_T_A*dk_T_AD)*X15, \
                    -k_a*A*X0 - k_a*dk_a_UDP*A*X1 - k_a*dk_a_TTP*A*X2 - k_a*dk_a_TDP*A*X3 + k_b*X4 + k_b*dk_b_UDP*X5 + k_b*dk_b_TTP*X6 + k_b*dk_b_TDP*X7 - k_a*dk_a_STP*A*X8 - k_a*dk_a_SDP*A*X9 - k_a*dk_a_DTP*A*X10 - k_a*dk_a_DDP*A*X11 + k_b*dk_b_STP*X12 + k_b*dk_b_SDP*X13 + k_b*dk_b_DTP*X14 + k_b*dk_b_DDP*X15, \
                    k_h*X0 + k_h*dk_h_AU*X4 + k_h*dk_h_T*X2 + k_h*dk_h_T*dk_h_AT*X6 + k_h*dk_h_S*X8 + k_h*dk_h_S*dk_h_AS*X12 + k_h*dk_h_D*X10 + k_h*dk_h_D*dk_h_AD*X14])
    
    return rate

########################################
###### Check for typos in equations (conservation of mass, KaiC derivatives sum to 0)
########################################

def test_eqs_right(q):
    X0 = np.array([3.5/16.]*16 + [1.5] + [0.])
    test_t = np.linspace(0, 12*60*60, 1000)
    converted_q= converter(q, 0.5, 303.15)
    derivative= lambda u, t: ddt(u, t, converted_q)
    solver= odespy.odepack.Lsoda(derivative, adams_or_bdf= 'bdf')
    solver.set_initial_condition(X0)
    phi, t= solver.solve(test_t)
    if not np.allclose(sum(ddt(X0, 0, converted_q)[:-2]), 0):
        print(ddt(X0, 0, converted_q)[:-2])
        #print(np.sum(phi[:, :-1], 1))
        print("KaiC derivatives don't sum to 0 :(")
    if np.any(phi < 0):
        print('Negative concentrations!')
    if not np.allclose(np.sum(phi[:, :-2], 1), 3.5):
        #print(np.sum(phi[:, :-2], 1))
        print(phi[0:3])
        #print(ddt_kcka_kbs(phi[1], 0, q, 1)[:-1])
        print("total KaiC mass isn't constant through time :(")
    return None


walker= np.array([1]*43 + [0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
test_eqs_right(walker)

########################################
###### Model priors
########################################

# on and off rates from Kageyama et al. 2006, Mol. Cell.
# presumably the on and off rates of the dephosphorylated state
k_a_prior= k_a_exp= np.log10(0.0279) # unit: s^-1*uM^-1
k_b_prior= k_b_exp= np.log10(0.0663) # unit: s^-1

# on and off rates from Mori et al. 2018, Nat. Commun.
# based on the phosphomimetic measurements; all units are s^-1
k_b_D_exp= np.log10(1./0.26)
k_b_S_exp= np.log10(1./0.43)
k_b_T_exp= np.log10(1./1.0)

dk_b_DDP_prior= dk_b_DTP_prior= k_b_D_exp - k_b_exp
dk_b_TDP_prior= dk_b_TTP_prior= k_b_T_exp - k_b_exp
dk_b_SDP_prior= dk_b_STP_prior= k_b_S_exp - k_b_exp
    
k_priors= [0., 0., k_a_prior, k_b_prior, 0., 0.]
dk_priors= [0.]*10 + [0., 0., dk_b_TDP_prior, dk_b_SDP_prior, 0., dk_b_DDP_prior, dk_b_TTP_prior, dk_b_STP_prior, 0., dk_b_DTP_prior] + [0.]*14

# and an inverse-gamma(1, 0.01) distribution for global error sigma
priors = [stats.norm(loc= mean, scale= 3) for mean in k_priors] + [stats.laplace(loc= mean, scale= 1) for mean in dk_priors] + [stats.invgamma(1, scale= 0.01)]

# initial condition weight used for Dirichlet distribution
init_cond_vec = np.array([20.,100.,1.,1.,1.,1.,1.,1.])

########################################
###### Model priors
########################################

# finally make instance of Model, loading in all info defined above
mod = Model_Pi('ds', 
            param_names, formatted_param_names,
            state_vars, formatted_state_vars, 
            ddt, converter, priors,
            div_KaiA_by_2=True,
            estimate_init_conds=True, init_cond_vector=init_cond_vec,
            init_indices=init_indices)

