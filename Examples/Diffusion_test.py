#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:32:44 2020


@author: Vicky

Neural PDE - Tensorflow 1.14
Testing with Diffusion Equation (2D)

IC: 
BC: Zero-Flux Boundary Condition

"""
import time 
import numpy as np 
import tensorflow as tf 
from matplotlib import pyplot as plt 
from pyDOE import lhs

import os 
npde_path = os.path.abspath('..')
#npde_path = npde_path + '/Neural_PDE'

save_location = npde_path + '/Trained_Models/Diffusion'

print(npde_path)


import sys 
sys.path.insert(0, npde_path) 


import Neural_PDE as npde
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


@tf.function
@tf.autograph.experimental.do_not_convert
def pde_func(model, X):
    t = X[:, 0:1]
    x = X[:, 1:2]
    y = X[:, 2:3]
    
    u = model(tf.concat([t, x, y],1), training=True)

    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    print(u_xx)
    u_y = tf.gradients(u, y)[0]
    u_yy = tf.gradients(u_y, y)[0]

    pde_loss = u_t - 0.1*(u_xx + u_yy)

    return pde_loss

# %%

N_f = NPDE_parameters['N_domain']
N_i = NPDE_parameters['N_initial']
N_b = NPDE_parameters['N_boundary']

data = np.load(npde_path + '/Data/Diffusion_data_2d.npz')

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
X_f = lb + (ub-lb)*lhs(3, N_f) 



training_data = {'X_i': X_i.astype('float64'), 'u_i': u_i.astype('float64'),
                'X_b': X_b.astype('float64'), 'u_b': u_b.astype('float64'),
                'X_f': X_f.astype('float64')}

# %%

model = npde.main.setup(NN_parameters, NPDE_parameters, PDE_parameters, pde_func)

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

tf.saved_model.save(model.model, save_location)
print('\n')
total_time = time_GD + time_QN
print("Total Time : {}".format(total_time))
# %%

# model_location = npde_path + '/Trained_Models/Diffusion_Initial'
# trained_model = tf.saved_model.load(model_location)

# T_star = np.expand_dims(np.repeat(t, len(XY_star)), 1)
# X_star_tiled = np.tile(XY_star, (len(t), 1))

# X_star = np.hstack((T_star, X_star_tiled))

# u_pred = trained_model(X_star)
# u_pred = np.reshape(u_pred, np.shape(u))


# def moving_plot(u_actual, u_sol):
#     plt.figure()
    
#     for ii in range(len(t)):
#         plt.contourf(u_actual[ii], cmap='plasma')
#         plt.contourf(u_sol[ii], alpha=0.5)
#         plt.pause(0.01)
#         plt.clf()
        
# moving_plot(u, u_pred)
