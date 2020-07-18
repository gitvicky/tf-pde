#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:04:32 2020

@author: Vicky

Neural PDE - Tensorflow 2.X
Testing with Convection Diffusion Equation

PDE: u_t - 0.2*u_xx +0.1*u 
IC: u(0, x) = 4x - 4x^2,
BC: Periodic 
Domain: t ∈ [0,0.2],  x ∈ [0,1.0]
"""

import numpy as np 

import os 
npde_path = os.path.abspath('..')
#npde_path = npde_path + '/Neural_PDE'

import sys 
sys.path.insert(0, npde_path) 
    
import tfpde
# %%
#Neural Network Hyperparameters
NN_parameters = {'Network_Type': 'Regular',
                'input_neurons' : 2,
                'output_neurons' : 1,
                'num_layers' : 4,
                'num_neurons' : 64,
                }


#Neural PDE Hyperparameters
NPDE_parameters = {'Sampling_Method': 'Initial',
                   'N_initial' : 100, #Number of Randomly sampled Data points from the IC vector
                   'N_boundary' : 300, #Number of Boundary Points
                   'N_domain' : 5000 #Number of Domain points generated
                  }


#PDE 
PDE_parameters = {'Inputs': 't, x',
                  'Outputs': 'u',
                  'Equation': 'D(u, t) - 0.2*D2(u, x) - 0.1*u',
                  'lower_range': [0.0, 0.0], #Float 
                  'upper_range': [0.2, 1.0], #Float
                  'Boundary_Condition': "Dirichlet", #Periodic 
                  'Boundary_Vals' : None,
                  'Initial_Condition': lambda x: 4*x - 4*x**2,
                  'Initial_Vals': None
                 }

# %%
#Using Simulation Data at the Initial and Boundary Values (BC would be Dirichlet under that case )

N_f = NPDE_parameters['N_domain']
N_i = NPDE_parameters['N_initial']
N_b = NPDE_parameters['N_boundary']

# Data Location
data_loc = os.path.abspath('..') + '/Data/'
data = np.load(data_loc + 'ConvDiff_1D.npz')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]

Exact = np.real(data['U_sol']).T

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

X_f = tfpde.sampler.initial_sampler(N_f, lb, ub) #Factors generated using LHS 

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
'''
N_i = NPDE_parameters['N_initial']
N_b = NPDE_parameters['N_boundary']
N_f = NPDE_parameters['N_domain']

lb = PDE_parameters['lower_range']
ub = PDE_parameters['upper_range']

Initial_Condition = PDE_parameters['Initial_Condition']
Boundary_vals = PDE_parameters['Boundary_Vals']

X_i = tfpde.sampler.initial_sampler(N_i, lb, ub)
X_b = tfpde.sampler.boundary_sampler(N_b, lb, ub)
X_f = tfpde.sampler.domain_sampler(N_f, lb, ub)

u_i = Initial_Condition(X_i[:,1:2])
u_b = Boundary_vals

# X_i = 2.0*(X_i - np.asarray(lb))/(np.asarray(ub) - np.asarray(lb)) - 1.0
# X_b = 2.0*(X_b - np.asarray(lb))/(np.asarray(ub) - np.asarray(lb)) - 1.0
# X_f = 2.0*(X_f - np.asarray(lb))/(np.asarray(ub) - np.asarray(lb)) - 1.0

training_data = {'X_i': X_i, 'u_i': u_i,
                'X_b': X_b, 'u_b': u_b, 
                'X_f': X_f}
'''
# %%

model = tfpde.main.setup(NN_parameters, NPDE_parameters, PDE_parameters)

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
# Data Location
data_loc = os.path.abspath('..') + '/Data/'
data = np.load(data_loc + 'ConvDiff_1D.npz')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]

Exact = np.real(data['U_sol']).T

X, T = np.meshgrid(x,t)

X_star = np.hstack((T.flatten()[:,None], X.flatten()[:,None])) #Flattened array with the inputs  X and T 
u_star = Exact.flatten()[:,None]               

# X_star = 2.0*(X_star - np.asarray(lb))/(np.asarray(ub) - np.asarray(lb)) - 1.0

u_pred = model.predict(X_star)
u_pred = np.reshape(u_pred, np.shape(Exact))

tfpde.plotter.evolution_plot(Exact, u_pred)
    