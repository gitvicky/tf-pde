#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:32:44 2020


@author: Vicky

Neural PDE - Tensorflow 1.14
Testing with Diffusion Equation (2D)

PDE: u_t - 0.1(u_xx + u_yy)
IC: u(0, x) = 
BC: Neumann
Domain: t ∈ [0,10],  x ∈ [0,1], y ∈ [0,1]

"""


import os 
import numpy as np
import tfpde
# %%
#Neural Network Hyperparameters
NN_parameters = {
                'input_neurons' : 3,
                'output_neurons' : 1,
                'num_layers' : 2,
                'num_neurons' : 25,
                }


#Neural PDE Hyperparameters
NPDE_parameters = {'Sampling_Method': 'Random',
                   'N_initial' : 200, #Number of Randomly sampled Data points from the IC vector
                   'N_boundary' : 5000, #Number of Boundary Points
                   'N_domain' : 50000 #Number of Domain points generated
                  }


#PDE 
PDE_parameters = {'Inputs': 't, x',
                  'Outputs': 'u',
                  'Equation': 'D(u, t) - 0.1*(D2(u, x) + D2(u, y))',
                  'lower_range': [0.0, 0.0, 0.0], #Float 
                  'upper_range': [10.0, 1.0, 1.0], #Float
                  'Boundary_Condition': "Neumann",
                  'Boundary_Vals' : 0,
                  'Initial_Condition': None,
                  'Initial_Vals': None
                 }

# %%

N_f = NPDE_parameters['N_domain']
N_i = NPDE_parameters['N_initial']
N_b = NPDE_parameters['N_boundary']

# Data Location
data_loc = os.path.abspath('..') + '/Data/'
data = np.load(data_loc + 'diffusion_data_2d.npz')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
y = data['y'].flatten()[:,None]

u = np.real(data['U_sol'])

lb = np.asarray(PDE_parameters['lower_range'])
ub = np.asarray(PDE_parameters['upper_range'])


X, Y = np.meshgrid(x, y)
XY_star = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
X_IC = np.hstack(( np.zeros(len(XY_star)).reshape(len(XY_star), 1), XY_star))
u_IC = u[0].flatten()
u_IC = np.expand_dims(u_IC, 1)

X, T = np.meshgrid(x, t[1:])
XT_star = np.hstack((T.flatten()[:,None], X.flatten()[:,None]))
X_BC_y0 = np.insert(XT_star, 2, np.ones(len(XT_star)) * lb[1], axis=1)
u_BC_y0 = u[1:, 0, :].flatten()       

X_BC_yf = np.insert(XT_star, 2, np.ones(len(XT_star)) * ub[1] , axis=1)
u_BC_yf = u[1:, -1, :].flatten()   


Y, T = np.meshgrid(y, t[1:])
YT_star = np.hstack((T.flatten()[:,None], Y.flatten()[:,None]))
X_BC_x0 = np.insert(YT_star, 1, np.ones(len(YT_star)) * lb[0], axis=1)
u_BC_x0 = u[1:, :, 0].flatten()       

X_BC_xf = np.insert(YT_star, 1, np.ones(len(YT_star)) * ub[0] , axis=1)
u_BC_xf = u[1:, :, -1].flatten()  

X_BC = np.vstack((X_BC_y0, X_BC_yf, X_BC_x0, X_BC_xf))
u_BC = np.hstack((u_BC_y0, u_BC_yf, u_BC_x0, u_BC_xf))
u_BC = np.expand_dims(u_BC, 1)


#Selecting the specified number of Initial points. 
idx = np.random.choice(X_IC.shape[0], N_i, replace=False) 
X_i = X_IC[idx]
u_i = u_IC[idx]

#Selecting the specified number of Boundary points. 
idx = np.random.choice(X_BC.shape[0], N_b, replace=False)
X_b = X_BC[idx] 
u_b = u_BC[idx] 

#Obtaining the specified number of Domain Points using Latin Hypercube Sampling within lower and upper bounds. 
X_f = tfpde.sampler.initial_sampler(N_f, lb, ub)


training_data = {'X_i': X_i.astype('float64'), 'u_i': u_i.astype('float64'),
                'X_b': X_b.astype('float64'), 'u_b': u_b.astype('float64'),
                'X_f': X_f.astype('float64')}

# %%

model = tfpde.main.setup(NN_parameters, NPDE_parameters, PDE_parameters)

# %%
train_config = {'Optimizer': 'adam',
                 'learning_rate': 0.001, 
                 'Iterations' : 50000}

time_GD = model.train(train_config, training_data)

# %%
train_config = {'Optimizer': 'L-BFGS-B',
                 'learning_rate': None, 
                 'Iterations' : None}

time_QN = model.train(train_config, training_data)


# %%

# T_star = np.expand_dims(np.repeat(t, len(XY_star)), 1)
# X_star_tiled = np.tile(XY_star, (len(t), 1))

# X_star = np.hstack((T_star, X_star_tiled))

# u_pred = model(X_star)
# u_pred = np.reshape(u_pred, np.shape(u))


