#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:17:47 2020

@author: Vicky


Neural PDE - Tensorflow 2.X
Module : Model

Training Ground Class which houses all the associated training functions - loss functions, gradient functions, callbacks, training loops and evaluation functions 
"""
import time
import numpy as np
import tensorflow as tf

from .network import Network
from .pde import PDE
from . import boundary_conditions
from .sampler import Sampler
from . import options 
from . import qnw

class TrainingGround(Network, Sampler, PDE):
    
    def __init__(self, layers, lb, ub, activation, initializer, BC, BC_Vals, N_f, network_type, pde_func, eqn_str, in_vars, out_vars, sampler):
        """
        

        Parameters
        ----------
        layers : LIST
            Nunmber of neurons in each layer
        lb : ARRAY
            Lower Range of the time and space domain
        ub : ARRAY
            Upper Range of the time and space domain
        activation : STR
            Name of the activation Function
        initializer : STR
            Name of the Initialiser for the neural network weights
        N_f : INT
            Number of points sampled from the domain space.
        pde_func : FUNC
            Explcitly defined domain function.
         eqn_str : STR
            The PDE in string with the specified format.
        in_vars : INT
            Number of input variables.
        out_vars : INT
            Number of output variables.

        Returns
        -------
        None.

        """
        
        Network.__init__(self, layers, lb, ub, activation, initializer)
        Sampler.__init__(self, N_f, subspace_N = int(N_f/10))   #Percentage to be sammpled from the subspace. 
        PDE.__init__(self, eqn_str, in_vars, out_vars)
        
        self.layers = layers 
        self.input_size = self.layers[0]
        self.output_size = self.layers[-1]
        
        self.bc = boundary_conditions.select(BC)
        
        if network_type == 'Regular':
            self.model = Network.initialize_NN(self)
        elif network_type == 'Resnet':
            self.model = Network.initialize_resnet(self, num_blocks=2)
        else:
            raise ValueError("Unknown Network Type. It should be either 'Regular' or 'Resnet'")

        self.trainable_params = self.model.trainable_weights

        
        self.pde = PDE.func  #Implicit 
        # self.pde = pde_func #Explicit
        
        self.loss_list =[]
        self.sampler = sampler
        
    #Explicit
    # def pde_func(self, X):
    #     pde_loss = self.pde(self.model, X)
    #     return pde_loss


        
    def ic_func(self, X, u):
        u_pred = self.model(X, training=True)
        ic_loss = u_pred - u
        return ic_loss

    
    def bc_func(self, X, u):
        bc_loss = self.bc(self.model, X, u)
        return bc_loss
     
    #Implicit
    def pde_func(self, X):
        pde_loss = self.pde(self, X)
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
        """
        

        Parameters
        ----------
        X_i : NUMPY ARRAY
            Initial input points.
        u_i : NUMPY ARRAY
            Initial outputs.
        X_b : NUMPY ARRAY
            Boundary input points.
        u_b : NUMPY ARRAY
            Boundary outputs.
        X_f : NUMPY ARRAY
            Domain input points.

        Returns
        -------
        model_loss : TENSOR
            Sum of Initial, Boundary and Domain MSE loss
        model_gradients : TENSOR
            Loss gradient with respect to thge model trainable params. 

        """
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
            if self.sampler == 'Initial':
                nIter_2 = nIter
            else :
                nIter_2 = int(nIter/2)
            
            for it in range(nIter):
                
                model_loss, model_gradients = self.loss_and_gradients(X_i, u_i, X_b, u_b, X_f)
                optimizer.apply_gradients(zip(model_gradients, self.trainable_params))
                
                if it%10 == 0:
                    self.callback_GD(it, model_loss)
                
             # if self.sampler == 'Residual':   
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
                
            # elif self.sampler == 'Uniform':
                # X_f_sampled = Sampler.uniform_sampler(self)
        
                # for it in range(nIter_2, nIter):
    
                #     if it %500 ==0:
                #         X_f_sampled = Sampler.uniform_sampler(self)
                        
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


        elif kind == "QN_TFP":
            
            func = qnw.TFP_Keras_Wrapper(self.model, self.loss_func, X_i, u_i, X_b, u_b, X_f)
            # convert initial model parameters to a 1D tf.Tensor
            init_params = tf.dynamic_stitch(func.idx, self.model.trainable_variables)
            
            trained_variables = optimizer(value_and_gradients_function=func,
                                                   initial_position=init_params,
                                                   max_iterations=500)
            
            func.assign_new_model_parameters(trained_variables.position)
                    
        end_time = time.time() - start_time 
        return end_time
        
        
    def predict(self, X):
        return self.model(X).numpy()
    
    
    def retrain(self, model, train_config, train_data):
        self.model = model
        self.trainable_params = self.model.trainable_variables
        
        return self.train(train_config, train_data)