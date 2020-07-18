#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:13:13 2020

@author: Vicky

Neural PDE - Tensorflow 2.X
Module : Options
"""

import numpy as np
from scipy import optimize
import tensorflow as tf
import tensorflow_probability as tfp


# ------------------ OPTIMIZER ------------------------------


def get_optimizer(name, lr=None):
    
    if name in ["sgd", "nadam", "adagrad", "adadelta", "adamax", "adam", "rmsprop"]:
        return {
                "sgd": tf.keras.optimizers.SGD(lr),
                "nadam": tf.keras.optimizers.Nadam(lr),
                "adagrad": tf.keras.optimizers.Adagrad(lr),
                "adadelta": tf.keras.optimizers.Adadelta(lr),
                "adamax": tf.keras.optimizers.Adamax(lr),
                "adam": tf.keras.optimizers.Adam(lr),
                "rmsprop": tf.keras.optimizers.RMSprop(lr),
                }[name], "GD"
    
    elif name in ["BFGS", "L-BFGS"]:
            return {    
                    "BFGS": tfp.optimizer.bfgs_minimize,
                    "L-BFGS": tfp.optimizer.lbfgs_minimize,        
                    }[name], "QN_TFP"
    else:
        return  optimize.minimize, "QN_Scipy"
         
    
    raise ValueError("Unknown Optimizer")
        
    
    
    
# ------------------ ACTIVATION FUNCTION ----------------------------

    
def get_activation(name):
    return {
            "tanh": tf.tanh,
            "sigmoid": tf.sigmoid,
            "relu": tf.nn.relu,
            "leaky_relu": tf.nn.leaky_relu
            }[name]
    raise ValueError("Unknown Activation Function")
    
    
    
# ------------------ KERNEL INITIALIZER ----------------------------

    
def get_initializer(name):
    return {
            "Glorot Uniform": tf.keras.initializers.GlorotUniform(),
            "Glorot Normal": tf.keras.initializers.GlorotNormal(),
            "Random Normal": tf.keras.initializers.RandomNormal(),
            "Random Uniform": tf.keras.initializers.RandomUniform(),
            "Truncatd Normal": tf.keras.initializers.TruncatedNormal(),
            "Variance Scaling": tf.keras.initializers.VarianceScaling(),
            "Constant": tf.constant_initializer(value =1),
            "Zero": tf.zeros_initializer()
            }[name]
    raise ValueError("Unknown Initializer")