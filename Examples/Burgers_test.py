#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:04:32 2020

@author: Vicky

Neural PDE - Tensorflow 2.X
Testing with Burgers Equation

PDE: u_t + u*u_x - 0.1*u_xx
IC: u(0, x) = -sin(pi.x/8)
BC: Periodic 
Domain: t ∈ [0,10],  x ∈ [-8,8]
"""
import os 
import numpy as np 
import scipy.io

from pyDOE import lhs
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
                   'N_boundary' : 100, #Number of Boundary Points
                   'N_domain' : 5000 #Number of Domain points generated
                  }


#PDE 
PDE_parameters = {'Inputs': 't, x',
                  'Outputs': 'u',
                  'Equation': 'D(u, t) + u*D(u, x) - 0.1*D2(u, x)',
                  'lower_range': [0.0, -8.0], #Float 
                  'upper_range': [10.0, 8.0], #Float
                  'Boundary_Condition': "Dirichlet", #Periodic 
                  'Boundary_Vals' : None,
                  'Initial_Condition': lambda x: -np.sin((np.pi*x)/8),
                  'Initial_Vals': None
                 }


# %%

#Using Simulation Data at the Initial and Boundary Values (BC would be Dirichlet under that case )

N_f = NPDE_parameters['N_domain']
N_i = NPDE_parameters['N_initial']
N_b = NPDE_parameters['N_boundary']

# Data Location
data_loc = os.path.abspath('..') + '/Data/'
data = scipy.io.loadmat(data_loc +'burgers.mat')

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

X_f = tfpde.sampler.domain_sampler(N_f, lb, ub) 

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
data_loc = os.path.abspath('..') + '/Data/'
data = scipy.io.loadmat(data_loc +'burgers.mat')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x,t)

X_star = np.hstack((T.flatten()[:,None], X.flatten()[:,None])) 
u_star = Exact.flatten()[:,None]              

u_pred = model.predict(X_star)
u_pred = np.reshape(u_pred, np.shape(Exact))

tfpde.plotter.evolution_plot(Exact, u_pred)
    