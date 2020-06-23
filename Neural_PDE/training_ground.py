#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:17:47 2020

@author: Vicky


Neural PDE - Tensorflow 2.X
Module : Model
"""
import time
import numpy as np
import tensorflow as tf

from .network import Network
from . import boundary_conditions
from .sampler import Sampler
from . import options 
from . import qnw

class TrainingGround(Network, Sampler):
    def __init__(self, layers, lb, ub, activation, initializer, N_f, pde):
        
        Network.__init__(self, layers, lb, ub, activation, initializer)
        Sampler.__init__(self, N_f, subspace_N = int(N_f/10))   #Percentage to be sammpled from the subspace. 
        
        self.layers = layers 
        self.input_size = self.layers[0]
        self.output_size = self.layers[-1]
        
        self.bc = boundary_conditions.select('Dirichlet')
        self.pde = pde
        
        self.model = Network.initialize_NN(self)
        self.trainable_params = self.model.trainable_weights

        self.loss_list =[]
        
        
    def ic_func(self, X, u):
        u_pred = self.model(X, training=True)
        ic_loss = u_pred - u
        return ic_loss

    
    def bc_func(self, X, u):
        bc_loss = self.bc(self.model, X, u)
        return bc_loss
        
    def pde_func(self, X):
        pde_loss = self.pde(self.model, X)
        return pde_loss
    


    
    
    def loss_func(self, X_i, u_i, X_b, u_b, X_f):
        
        initial_loss = self.ic_func(X_i, u_i)
        boundary_loss = self.bc_func(X_b, u_b)
        domain_loss = self.pde_func(X_f)
        
        return tf.reduce_mean(tf.square(initial_loss)) + \
                            tf.reduce_mean(tf.square(boundary_loss)) + \
                            tf.reduce_mean(tf.square(domain_loss))
    
    @tf.function
    @tf.autograph.experimental.do_not_convert
    def loss_and_gradients(self, X_i, u_i, X_b, u_b, X_f):
        with tf.GradientTape() as tape:
            model_loss=self.loss_func(X_i, u_i, X_b, u_b, X_f)
        model_gradients = tape.gradient(model_loss, self.trainable_params)
        return model_loss, model_gradients
    
    def callback_GD(self, it, loss_value):
        elapsed = time.time() - self.init_time
        self.loss_list.append(loss_value)
        print('GD.  It: %d, Loss: %.3e, Time: %.2f' % 
                  (it, loss_value, elapsed))
        self.init_time = time.time()
        
    
    
    def train(self, train_config, train_data):
        start_time = time.time()
    
        optimizer, kind = options.get_optimizer(name=train_config['Optimizer'], lr=train_config['learning_rate'])
        nIter = train_config['Iterations']
        
        
        X_i = train_data['X_i']
        u_i = train_data['u_i']
        X_b = train_data['X_b']
        u_b = train_data['u_b']
        X_f = train_data['X_f']
                
        
        self.init_time = time.time()
        
        if kind == "GD":
        
            nIter_2 = int(nIter/2)
            
            for it in range(nIter):
                
                model_loss, model_gradients = self.loss_and_gradients(X_i, u_i, X_b, u_b, X_f)
                optimizer.apply_gradients(zip(model_gradients, self.trainable_params))
                
                if it%10 == 0:
                    self.callback_GD(it, model_loss)
                
                
            # X_f_sampled = Sampler.str_sampler(self)
            # # X_f_sampled = tf.concat([X_f, X_f_sampled], axis=0)
            # X_f_sampled = np.vstack((X_f, X_f_sampled))
    
            # for it in range(nIter_2, nIter):

            #     if it %500 ==0:
            #         X_f_sampled = Sampler.str_sampler(self)
            #         # X_f_sampled = tf.concat([X_f, X_f_sampled], axis=0)
            #         X_f_sampled = np.vstack((X_f, X_f_sampled))
                    
            #     model_loss, model_gradients = self.loss_and_gradients(X_i, u_i, X_b, u_b, X_f_sampled)
            #     optimizer.apply_gradients(zip(model_gradients, self.trainable_params))
                
            #     if it%10 == 0:
            #         self.callback_GD(it, model_loss)

                    
        elif kind == "QN_Scipy":
            
            func = qnw.Scipy_Keras_Wrapper(self.model, self.loss_func, X_i, u_i, X_b, u_b, X_f)
            # convert initial model parameters to a 1D tf.Tensor
            init_params = tf.dynamic_stitch(func.idx, self.model.trainable_variables)
            
            trained_variables = optimizer(fun=func,
                                          x0=init_params,
                                          jac=True,
                                          method = train_config['Optimizer'])
            
            func.assign_new_model_parameters(trained_variables.x)

            
                    
            
        
        print("Total Training Time : {}".format(time.time() - start_time))
        
        
    def predict(self, X):
        return self.model(X).numpy()
