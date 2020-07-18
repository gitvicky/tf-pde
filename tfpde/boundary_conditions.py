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
    try: 
        return {
            "Dirichlet": dirichlet,
            "Neumann": neumann,
            "Periodic": periodic
            }[name]
    except KeyError:
        raise KeyError("Unknown Boundary Condition")


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
    t = X[:, 0:1]
    x = X[:, 1:2]
    n = int(X.shape[0]/2)
    u =  model(tf.concat([t, x], 1), training=True)
    u_x = tf.gradients(u, x)[0]
    
    return (u[:n] - u[n:]) + (u_x[:n] - u_x[:n])

