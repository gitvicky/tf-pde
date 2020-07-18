#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:59:18 2020

@author: Vicky

Neural PDE - Tensorflow 2.X
Module : Main


Basefile which takes in the input parameters that define the PDE, NN and NPDE, extracts the relevant information
from those dicts and is send them to set up ther network and the training graphs. 

"""

import numpy as np

from . import training_ground

def setup(NN, NPDE, PDE, pde_func):
    """
    

    Parameters
    ----------
    NN : DICT
        Neural Network Architecture Parameters.
    NPDE : DICT
        Sampling Methods and Points to be sampled. 
    PDE : DICT
       PDE in string, input parameters, output parameters, IC, BC, lower and upper ranges
    pde_func : FUNC
        Explcitly defined domain function.

    Returns
    -------
    model : OBJECT
        Object of the class Training Ground instantiated with the defined input parameters. 

    """
    
    N_f = NPDE['N_domain']
    sampler = NPDE['Sampling_Method']
    
    network_type = NN['Network_Type']
    input_size = NN['input_neurons']
    num_layers = NN['num_layers']
    num_neurons = NN['num_neurons']
    output_size = NN['output_neurons']
    
    eqn_str = PDE['Equation']
    input_str = PDE['Inputs']
    output_str = PDE['Outputs']
    lb = np.asarray(PDE['lower_range'])
    ub = np.asarray(PDE['upper_range'])
    BC = PDE['Boundary_Condition']
    BC_Vals = PDE['Boundary_Vals']

    layers = np.concatenate([[input_size], num_neurons*np.ones(num_layers), [output_size]]).astype(int).tolist() 
    
    
    activation = 'tanh'
    initialiser = 'Glorot Uniform'

    
    model = training_ground.TrainingGround(layers,
                                         lb, ub,
                                         activation, 
                                         initialiser,
                                         BC, 
                                         BC_Vals,
                                         N_f,
                                         network_type,
                                         pde_func,
                                         eqn_str,
                                         input_str,
                                         output_str,
                                         sampler)
            
    
    return model