#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:02:09 2020

Neural PDE - Tensorflow 2.X
Module : Spatio-Temporal Residual Sampler
"""
import numpy as np
import tensorflow as tf 
from pyDOE import lhs


class Sampler(object):
    
    def __init__(self, N_samples, subspace_N):

        self.N = N_samples
        self.ssN = subspace_N
        
    def residual(self, n, t_bounds):
        residual_across_time = []
        
        for ii in range(n-1):
            lb_temp = np.asarray([t_bounds[ii], self.lb[1]])
            ub_temp = np.asarray([t_bounds[ii+1], self.ub[1]])
            X_f = lb_temp + (ub_temp-lb_temp)*lhs(self.input_size, self.ssN) 
            
            str_val = tf.reduce_mean(tf.square(self.pde_func(X_f)))
            residual_across_time.append(str_val)

        return np.asarray(residual_across_time)

        
    def str_sampler(self):
        n = int(self.N/self.ssN)
        t_bounds = np.linspace(self.lb[0], self.ub[0], n)
        
        residuals = self.residual(n, t_bounds)
        t_range_idx = np.argmax(residuals)
        print('\n')
        print("Time_Index : {}".format(t_range_idx))
        
        lb_temp = np.asarray([t_bounds[t_range_idx], self.lb[1]])
        ub_temp = np.asarray([t_bounds[t_range_idx+1], self.ub[1]])
        X_f = lb_temp + (ub_temp-lb_temp)*lhs(self.input_size, self.ssN) 
        
        return X_f
        
    
    def uniform_sampler(self):
        X_f = self.lb + (self.ub-self.lb)*lhs(self.input_size, self.N) 
        return X_f
