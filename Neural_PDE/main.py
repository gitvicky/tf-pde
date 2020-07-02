#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:59:18 2020

@author: Vicky

Neural PDE - Tensorflow 2.X
Module : Main
"""

import numpy as np

from . import training_ground

def setup(NN, NPDE, PDE, pde_func):
    
    N_f = NPDE['N_domain']
    
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

    layers = np.concatenate([[input_size], num_neurons*np.ones(num_layers), [output_size]]).astype(int).tolist() 
    
    
    activation = 'tanh'
    initialiser = 'Glorot Uniform'

    
    model = training_ground.TrainingGround(layers,
                                         lb, ub,
                                         activation, 
                                         initialiser,
                                         N_f,
                                         pde_func,
                                         eqn_str,
                                         input_str,
                                         output_str)
            
    
    return model