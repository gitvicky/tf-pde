#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:04:32 2020

@author: Vicky

Neural PDE - Tensorflow 1.14
Testing with Burgers 
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

save_location = npde_path + '/Trained_Models/Burgers'
    
import Neural_PDE as npde
# %%
#Neural Network Hyperparameters
NN_parameters = {
                'input_neurons' : 2,
                'output_neurons' : 1,
                'num_layers' : 4,
                'num_neurons' : 64,
                }


#Neural PDE Hyperparameters
NPDE_parameters = {'Sampling_Method': 'Random',
                   'N_initial' : 100, #Number of Randomly sampled Data points from the IC vector
                   'N_boundary' : 100, #Number of Boundary Points
                   'N_domain' : 5000 #Number of Domain points generated
                  }


#PDE 
PDE_parameters = {'Inputs': 't, x',
                  'Outputs': 'u',
                  'Equation': 'D(u, t) + u*D(u, x) - 0.1*D2(u, x)',
                  'lower_range': [0.0, -8.0], #Float 
                  'upper_range': [10.0, 8.0], #Float
                  'Boundary_Condition': "Dirichlet",
                  'Boundary_Vals' : None,
                  'Initial_Condition': lambda x: -np.sin((np.pi*x)/8),
                  'Initial_Vals': None
                 }

@tf.function
@tf.autograph.experimental.do_not_convert
def pde_func(model, X):
    t = X[:, 0:1]
    x = X[:, 1:2]
    
    u = model(tf.concat([t,x], 1), training=True)
    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]

    pde_loss = u_t + u*u_x - 0.1*u_xx
            
    # u = model(X, training=True)
    # u_X = tf.gradients(u, X)[0]
    # u_XX = tf.gradients(u_X, X)[0]
        
    # pde_loss = u_X[:, 0:1] + u*u_X[:, 1:2] - 0.1*u_XX[:, 1:2]

    return pde_loss

# %%

#Using Simulation Data at the Initial and Boundary Values (BC would be Dirichlet under that case )

N_f = NPDE_parameters['N_domain']
N_i = NPDE_parameters['N_initial']
N_b = NPDE_parameters['N_boundary']

data = scipy.io.loadmat(npde_path + '/Data/burgers.mat')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x,t)

X_star = np.hstack((T.flatten()[:,None], X.flatten()[:,None])) 
u_star = Exact.flatten()[:,None]              

# Domain bounds
lb = X_star.min(0) 
ub = X_star.max(0)
    
X_i = np.hstack((T[0:1,:].T, X[0:1,:].T))
u_i = Exact[0:1,:].T

X_lb = np.hstack((T[:,0:1], X[:,0:1])) 
u_lb = Exact[:,0:1] 
X_ub = np.hstack((T[:,-1:], X[:,-1:])) 
u_ub = Exact[:,-1:] 

u_lb = np.zeros((len(u_lb),1))
u_ub = np.zeros((len(u_ub),1))  

X_b = np.vstack((X_lb, X_ub))
u_b = np.vstack((u_lb, u_ub))

X_f = lb + (ub-lb)*lhs(2, N_f) 

idx = np.random.choice(X_i.shape[0], N_i, replace=False)
X_i = X_i[idx, :]
u_i = u_i[idx,:]

idx = np.random.choice(X_b.shape[0], N_b, replace=False)
X_b = X_b[idx, :] 
u_b = u_b[idx,:]



training_data = {'X_i': X_i, 'u_i': u_i,
                'X_b': X_b, 'u_b': u_b,
                'X_f': X_f}

# %%
'''
N_i = NPDE_parameters['N_initial']
N_b = NPDE_parameters['N_boundary']
N_f = NPDE_parameters['N_domain']

lb = PDE_parameters['lower_range']
ub = PDE_parameters['upper_range']

Initial_Condition = PDE_parameters['Initial_Condition']
Boundary_vals = PDE_parameters['Boundary_Vals']

X_i = npde.sampler.initial_sampler(N_i, lb, ub)
X_b = npde.sampler.boundary_sampler(N_b, lb, ub)
X_f = npde.sampler.domain_sampler(N_f, lb, ub)

u_i = Initial_Condition(X_i[:,1:2])
u_b = Boundary_vals

X_i = 2.0*(X_i - np.asarray(lb))/(np.asarray(ub) - np.asarray(lb)) - 1.0
X_b = 2.0*(X_b - np.asarray(lb))/(np.asarray(ub) - np.asarray(lb)) - 1.0
X_f = 2.0*(X_f - np.asarray(lb))/(np.asarray(ub) - np.asarray(lb)) - 1.0

training_data = {'X_i': X_i, 'u_i': u_i,
                'X_b': X_b, 'u_b': u_b,
                'X_f': X_f}
'''
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
data = scipy.io.loadmat(npde_path + '/Data/burgers.mat')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x,t)

X_star = np.hstack((T.flatten()[:,None], X.flatten()[:,None])) 
u_star = Exact.flatten()[:,None]              

# X_star = 2.0*(X_star - np.asarray(lb))/(np.asarray(ub) - np.asarray(lb)) - 1.0

u_pred = model.predict(X_star)
u_pred = np.reshape(u_pred, np.shape(Exact))

npde.plotter.evolution_plot(Exact, u_pred)
