# -*- coding: utf-8 -*-
"""
This file defines the Model class, the template for all model iterations
A desired model (for instance, double site with nucleotide-dependent KaiA binding terms)
    would be an instance of this Model class, with parameter names, ODEs, and equations
    defined in its own file
May 2018
"""

import numpy as np
from scipy import stats
import pkg_resources

# total concentation of KaiC
KaiC = 3.5

class Model(object):
    '''
    This class holds in all the information for a model, including:
        model name,
        parameter names (both simple and LaTeX versions),
        priors for model parameters,
        KaiC state names (both simple and LaTeX versions),
        ODE equations,
        initial condition settings
            (if they are estimated as MCMC parameters, a Dirichlet prior is 
            assumed, with necessary weight vector as input)
            
    With this information, this class can:
        generate a walker with model_instance.generate_walker() using the 
            model parameter priors and initial conditions
        compute the log prior of an inputted walker 
            with model_instance.calc_log_prior()
        draw initial conditions to start integration at t = 0 (draw_t0)
    '''
    
    # necessary information for initialization
    def __init__(self, name, 
                 model_param_names, formatted_model_param_names, 
                 state_var_names, formatted_state_var_names,
                 d_dt, converter, model_param_priors, 
                 KaiC=3.5, div_KaiA_by_2=True,
                 estimate_init_conds=False, init_cond_vector=[], init_indices=[]):
        
        # model name
        self.model_name = name
        
        # model parameters
        self.model_params = model_param_names
        self.formatted_model_params = formatted_model_param_names
        self.n_model_params = len(self.model_params)
        
        # dictionaries mapping parameter names to indices
        self.i_to_param = dict(enumerate(self.model_params))
        self.param_to_i = {i[1]:i[0] for i in self.i_to_param.items()}
        
        # model parameter priors        
        self.model_param_priors = model_param_priors
        
        # KaiC state variables
        self.state_vars = state_var_names
        self.formatted_state_vars = formatted_state_var_names
        self.n_state_vars = len(self.state_vars)
        
        
        # dictionaries mapping state names to indices, consistent with diagram
        self.i_to_sv = dict(enumerate(self.state_vars))
        self.sv_to_i = {i[1]:i[0] for i in self.i_to_sv.items()}
        
        
        # ODE equation
        self.d_dt = d_dt
        self.converter= converter

        # initial condition settings
        self.estimate_init_conds = estimate_init_conds
        self.init_cond_vector = init_cond_vector        
        self.init_conds = []
        self.init_indices = init_indices
        if self.estimate_init_conds:
            self.init_conds = np.array(self.state_vars)[self.init_indices]
        self.n_init_conds = len(self.init_conds)
        
        self.tot_params = self.model_params + list(self.init_conds[:-1])
        self.n_tot_params = self.n_model_params + self.n_init_conds - 1
        
        self.i_to_tot_params = dict(enumerate(self.tot_params))
        self.tot_params_to_i = {i[1]:i[0] for i in self.i_to_tot_params.items()}
        
        self.KaiC = KaiC
        self.div_KaiA_by_2 = div_KaiA_by_2
        
    # print some model information
    def __str__(self):
        return "Model: {} \n".format(self.model_name) + \
                "State variables: \n" + \
                "\t{}\n".format(self.state_vars) + \
                "{} total parameters: \n".format(self.n_tot_params) + \
                "\t {} model params:\n".format(self.n_model_params) + \
                "\t \t {}\n".format(self.model_params) + \
                "\t {} estimated initial conditions \n".format(self.n_init_conds)
                
    def __repr__(self):
        return str(self)
    
    # using the model parameter priors, outputs log prior
    # if the walker uses estimated initial conditions, returning -np.inf for any negative concentrations
    def calc_log_prior(self, q):
        lp = 0.
        
        if self.estimate_init_conds:
            c = q[self.n_model_params:]
            init_concs = np.concatenate((c, [KaiC - np.sum(c)]))
            if np.any(init_concs < 0):
                return -np.inf
            else:
                c_priors= stats.dirichlet(self.init_cond_vector)
                lp += c_priors.logpdf(init_concs/KaiC)
        
        for i in range(len(self.model_params)):
            lp += self.model_param_priors[i].logpdf(q[i])
        return lp
    
    # draw a Dirichlet vector using the underlying vector
    def generate_init_conds(self):
        return KaiC*np.random.dirichlet(self.init_cond_vector)
        
    # returns a walker from the priors
    def generate_walker(self):
        results= [p.rvs(size=1)[0] for p in self.model_param_priors[:self.n_model_params]]
        return np.concatenate((np.asarray(results), self.generate_init_conds()[:-1]))

    # returns t=0 vector of state variables to start integration
    def draw_t0(self, q, KaiA0):
        dim = len(q)
        c = q[(dim-self.n_init_conds+1):]
        c = np.concatenate((c, [KaiC - sum(c)]))
        X0 = np.zeros(self.n_state_vars)
        for i in range(len(self.init_indices)):
            X0[self.init_indices[i]] = c[i]
        X0[-1] = KaiA0
        return X0

class Model_Pi(Model):
    '''
    This class allows explicit tracking of Pi in solution
    '''
    def draw_t0(self, q, KaiA0):
        dim = len(q)
        c = q[(dim-self.n_init_conds+1):]
        c = np.concatenate((c, [KaiC - sum(c)]))
        X0 = np.zeros(self.n_state_vars)
        for i in range(len(self.init_indices)):
            X0[self.init_indices[i]] = c[i]
        X0[-2] = KaiA0
        return X0

