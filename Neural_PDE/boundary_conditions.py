#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:22:29 2020

@author: Vicky


Neural PDE - Tensorflow 2.X
Module : Boundary Conditions 
"""

import tensorflow as tf 

def select(name):
    return {
        "Dirichlet": dirichlet,
        "Neumann": neumann,
        "Periodic": periodic
    }[name]

@tf.function
@tf.autograph.experimental.do_not_convert
def dirichlet(model, X, u):
    u_pred = model(X, training=True)
    return u - u_pred

@tf.function
@tf.autograph.experimental.do_not_convert
def neumann(model, X, f): #Currently only for 1D
    u = model(X, training=True)
    u_X = tf.gradients(u, X)[0]
    
    return u_X[:, 1:2] - f

@tf.function
@tf.autograph.experimental.do_not_convert
def periodic(model, X, f): # Currently for only 1D
    n = int(len(X)/2)
    u =  model(X, training=True)
    u_X = tf.gradients(u, X)[0]
    u_XX = tf.gradients(u_X, X)[0]
    
    return (u[:n] - u[n:]) + (u_X[:,1:2][:n] - u_X[:,1:2][:n]) + (u_XX[:,1:2][:n] - u_XX[:,1:2][:n])

