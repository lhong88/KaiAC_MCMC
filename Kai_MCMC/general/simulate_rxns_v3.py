# -*- coding: utf-8 -*-
"""
defines functions that perform integration of model-specified 
ODES under desired conditions (for instance, to simulate an 
autophosphorylation reaction, dephosphorylation reaction, etc.)
"""

import numpy as np
#from scipy import integrate
import odespy
import sys
from time import time

from ..running_MCMC.print_utils import stdout_redirected # ignores integration warnings

ode_method= 'bdf'
nsteps= 1000

def simulate_autophos(subdat, q, mod, do_pool, fine_int=False, fine_dt=60):    
   	# subdat is a subset of an experiment data table with a certain (ATP, KaiA)
   	# condition, often called by 
   	# 	subdat = autophos_table[autophos_table.rxn == rxn], 
   	# where rxn is a unique code corresponding to an (ATP, KaiA) reaction.

    # convert hours to seconds
    auto_t = np.array(60.*60.*subdat.real_time)

    # (useful for plotting smooth trajectories)
    if fine_int:
        auto_t = np.arange(auto_t[0], auto_t[-1]+fine_dt, fine_dt)

    # extract ATP, KaiA
    rxn_ATP = subdat.ATP.iloc[0] / 100.; rxn_KaiA = subdat.KaiA.iloc[0]
    # divide KaiA by 2 if specifid in model
    if mod.div_KaiA_by_2:
        rxn_KaiA /= 2.
    
    # extract temperature (in Kelvin)
    rxn_temp= subdat.Temp.iloc[0] + 273.15
    
    # set initial conditions
    X0 = mod.draw_t0(q, rxn_KaiA)
    
    rates= mod.converter(q, rxn_ATP, rxn_temp)
    
    # integrate model
    derivative= lambda u, t: mod.d_dt(u, t, rates)
    solver= odespy.odepack.Lsoda(derivative, adams_or_bdf= ode_method, nsteps= nsteps)
    solver.set_initial_condition(X0)
    
    if do_pool:
        with stdout_redirected():
            try:
                time0= time()
                Xs, t= solver.solve(auto_t)
                time1= time()
            except:
                return False
            
            int_time= time1 - time0
    else:
        try:
            time0= time()
            Xs, t= solver.solve(auto_t)
            time1= time()
        except:
            return False
            
        int_time= time1 - time0

    return auto_t, Xs, int_time
    
def simulate_autophos_ss(subdat, q, mod, do_pool, fine_int=False, fine_dt=60):    
   	# subdat is a subset of an experiment data table with a certain (ATP, KaiA)
   	# condition, often called by 
   	# 	subdat = autophos_table[autophos_table.rxn == rxn], 
   	# where rxn is a unique code corresponding to an (ATP, KaiA) reaction.

    # convert hours to seconds
    auto_t = np.concatenate((np.array([0]), np.array(60.*60.*subdat.real_time)))

    # (useful for plotting smooth trajectories)
    if fine_int:
        auto_t = np.arange(auto_t[0], auto_t[-1]+fine_dt, fine_dt)

    # extract ATP, KaiA
    rxn_ATP = subdat.ATP.iloc[0] / 100.; rxn_KaiA = subdat.KaiA.iloc[0]
    # divide KaiA by 2 if specifid in model
    if mod.div_KaiA_by_2:
        rxn_KaiA /= 2.
    
    # extract temperature (in Kelvin)
    rxn_temp= subdat.Temp.iloc[0] + 273.15
    
    # set initial conditions
    X0 = mod.draw_t0(q, rxn_KaiA)
    
    rates= mod.converter(q, rxn_ATP, rxn_temp)
    
    # check if the data is from single-site S431A mutant
    rxn_name= subdat.rxn.iloc[0]
    if rxn_name.startswith('T'):
        rates_ind= [16, 17, 18, 19] # corresponding to dk_p_US, dk_d_SU, dk_p_TD, dk_d_DT
        rxn_KaiC= subdat.KaiC.iloc[0]
        rates[rates_ind]= 0.
        X0= np.array([0., rxn_KaiC, 0., 0.,
                      0., 0., 0., 0.,
                      0., 0., 0., 0.,
                      0., 0., 0., 0.,
                      rxn_KaiA, 0.])
    
    # integrate model
    derivative= lambda u, t: mod.d_dt(u, t, rates)
    solver= odespy.odepack.Lsoda(derivative, adams_or_bdf= ode_method, nsteps= nsteps)
    solver.set_initial_condition(X0)
    
    if do_pool:
        with stdout_redirected():
            try:
                time0= time()
                Xs, t= solver.solve(auto_t)
                time1= time()
            except:
                return False
            
            int_time= time1 - time0
    else:
        try:
            time0= time()
            Xs, t= solver.solve(auto_t)
            time1= time()
        except:
            return False
            
        int_time= time1 - time0

    return np.delete(auto_t, 0), np.array([Xs[-1]]), int_time


def simulate_dephos(dephospho_data, q, mod, do_pool, fine_int=False, fine_dt=60):    
    '''
    A slight modification of the dephos function to account for the Pi state variable.
    '''
    phos_t = 60*60*np.arange(-20,0,4)
    dephos_t = 60*60*np.array(dephospho_data.Time) 
    
    if fine_int:
        phos_t = np.arange(phos_t[0], phos_t[-1]+fine_dt, fine_dt)
        dephos_t = np.arange(dephos_t[0], dephos_t[-1]+fine_dt, fine_dt)
    
    rKaiC = 3.4
    ATP0 = 1.; KaiA0 = 1.3
    if mod.div_KaiA_by_2:
        KaiA0 /= 2.
    
    X0 = [rKaiC] + [0.]*15 + [KaiA0] + [0.]
    
    # extract temperature (in Kelvin)
    rxn_temp= 303.15
    
    rates= mod.converter(q, ATP0, rxn_temp)
    
    der_phos= der_dephos= lambda u, t: mod.d_dt(u, t, rates)
    solver_phos= odespy.odepack.Lsoda(der_phos, adams_or_bdf= ode_method, nsteps= nsteps)
    solver_dephos= odespy.odepack.Lsoda(der_dephos, adams_or_bdf= ode_method, nsteps= nsteps)
    
    solver_phos.set_initial_condition(X0)
    
    if do_pool:

        with stdout_redirected():
            try:
                Xs_auto, t= solver_phos.solve(phos_t)
            except:
                return False
            # start at last endpoint, removing ALL KAIA as well as Pi
            Xd0 = Xs_auto[-1, :]
            Xd0[[4,5,6,7,12,13,14,15,16,17]] = 0
            
            solver_dephos.set_initial_condition(Xd0)
            try:
                time0= time()
                Xs_dephos, t= solver_dephos.solve(dephos_t)
                time1= time()
            except:
                return False
            
            int_time= time1 - time0
    else:
        try:
            Xs_auto, t= solver_phos.solve(phos_t)
        except:
            return False
        # start at last endpoint, removing ALL KAIA
        Xd0 = Xs_auto[-1, :]
        Xd0[[4,5,6,7,12,13,14,15,16,17]] = 0
        
        solver_dephos.set_initial_condition(Xd0)
        try:
            time0= time()
            Xs_dephos, t= solver_dephos.solve(dephos_t)
            time1= time()
        except:
            return False

        int_time= time1 - time0
        
    return phos_t, Xs_auto, dephos_t, Xs_dephos, int_time

