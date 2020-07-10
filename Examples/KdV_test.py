#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:37:25 2020

@author: Vicky


Neural PDE - Tensorflow 2.X
Testing with Korteweg de Vries Equation
"""

import numpy as np 
import tensorflow as tf 
from matplotlib import pyplot as plt 
import scipy.io
from pyDOE import lhs

import os 
npde_path = os.path.abspath('..')
#npde_path = npde_path + '/Neural_PDE'

import sys 
sys.path.insert(0, npde_path) 


import Neural_PDE as npde
# %%
#Neural Network Hyperparameters
NN_parameters = {
                'input_neurons' : 2,
                'output_neurons' : 1,
                'num_layers' : 4,
                'num_neurons' : 64
                }


#Neural PDE Hyperparameters
NPDE_parameters = {'Sampling_Method': 'Random',
                   'N_initial' : 300, #Number of Randomly sampled Data points from the IC vector
                   'N_boundary' : 300, #Number of Boundary Points
                   'N_domain' : 20000 #Number of Domain points generated
                  }


#PDE 
PDE_parameters = {'Inputs': 't, x',
                  'Outputs': 'u',
                  'Equation': 'D(u, t) + u*D(u, x) + 0.0025*D3(u, x)',
                  'lower_range': [0.0, -1.0], #Float 
                  'upper_range': [1.0, 1.0], #Float
                  'Boundary_Condition': "Dirichlet",
                  'Boundary_Vals' : None,
                  'Initial_Condition': None,
                  'Initial_Vals': None
                 }

@tf.function
@tf.autograph.experimental.do_not_convert
def pde_func(model, X):
    t = X[:, 0:1]
    x = X[:, 1:2]
        
    u = model(tf.concat([t,x],1), training=True)
    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_xxx = tf.gradients(u_xx, x)[0]

    pde_loss = u_t + u*u_x + 0.0025*u_xxx
    
    return pde_loss

# %%

N_f = NPDE_parameters['N_domain']
N_i = NPDE_parameters['N_initial']
N_b = NPDE_parameters['N_boundary']

data = scipy.io.loadmat(npde_path + '/Data/KdV.mat')

t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['uu']).T

X, T = np.meshgrid(x,t)

X_star = np.hstack((T.flatten()[:,None], X.flatten()[:,None])) #Flattened array with the inputs  X and T 
u_star = Exact.flatten()[:,None]              

# Domain bounds
lb = X_star.min(0) #Lower bounds of x and t 
ub = X_star.max(0) #Upper bounds of x and t
    
X_i = np.hstack((T[0:1,:].T, X[0:1,:].T)) #Initial condition value of X (x=-1....1) and T (t = 0) 
u_i = Exact[0:1,:].T #Initial Condition value of the field u

X_lb = np.hstack((T[:,0:1], X[:,0:1])) #Lower Boundary condition value of X (x = -1) and T (t = 0...0.99)
u_lb = Exact[:,0:1] #Bound Condition value of the field u at (x = 11) and T (t = 0...0.99)
X_ub = np.hstack((T[:,-1:], X[:,-1:])) #Uppe r Boundary condition value of X (x = 1) and T (t = 0...0.99)
u_ub = Exact[:,-1:] #Bound Condition value of the field u at (x = 11) and T (t = 0...0.99)

X_b = np.vstack((X_lb, X_ub))
u_b = np.vstack((u_lb, u_ub))

X_f = lb + (ub-lb)*lhs(2, N_f) #Factors generated using LHS 

idx = np.random.choice(X_i.shape[0], N_i, replace=False)
X_i = X_i[idx, :] #Randomly Extract the N_u number of x and t values. 
u_i = u_i[idx,:] #Extract the N_u number of field values 

idx = np.random.choice(X_b.shape[0], N_b, replace=False)
X_b = X_b[idx, :] #Randomly Extract the N_u number of x and t values. 
u_b = u_b[idx,:] #Extract the N_u number of field values 



training_data = {'X_i': X_i, 'u_i': u_i,
                'X_b': X_b, 'u_b': u_b,
                'X_f': X_f}

# %%

model = npde.main.setup(NN_parameters, NPDE_parameters, PDE_parameters, pde_func)

# %%
train_config = {'Optimizer': 'adam',
                 'learning_rate': 0.001, 
                 'Iterations' : 5000}

time_GD = model.train(train_config, training_data)

# %%
train_config = {'Optimizer': 'L-BFGS-B',
                 'learning_rate': None, 
                 'Iterations' : None}

time_QN = model.train(train_config, training_data)
# %%

u_pred = model.predict(X_star)
u_pred = np.reshape(u_pred, np.shape(Exact))


npde.plotter.evolution_plot(Exact, u_pred)
