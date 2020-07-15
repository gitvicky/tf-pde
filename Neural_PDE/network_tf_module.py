#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:01:56 2020

@author: Vicky

Neural PDE - Tensorflow 2.X
Module : Network with tf.Module
"""

import numpy as np
import tensorflow as tf 
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)

tf.keras.backend.set_floatx('float64')

from . import options

class Network(tf.Module):
    def __init__(self, layers, lb, ub, activation, initializer, num_blocks=None, name=None):
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
        num_blocks : INT, optional
            If invoking a resnet with skip connections, the number of blocks required. The default is None.

        Returns
        -------
        None.

        """
        # self.layers = layers
        self.num_inputs = layers[0]
        self.num_outputs = layers[-1]
        self.neurons = layers[1]
        self.num_blocks = num_blocks

        self.lb = lb 
        self.ub = ub 
        
        self.activation_name = activation
        self.activation = options.get_activation(activation)
        self.initializer = options.get_initializer(initializer)
        
        with self.name_scope:
            self.layers.append(keras.layers.Dense(self.layers[1] , input_shape=(self.layers[0],),
                                     activation=self.activation,
                                     kernel_initializer=self.initializer))
            
            for ii in range(2, len(self.layers) - 2):
                self.layers.append(keras.layers.Dense(units = self.layers[ii],
                                activation = self.activation,
                                kernel_initializer = self.initializer
                                ))
            self.layers.append(keras.layers.Dense(units = self.layers[-1],
                activation = None,
                kernel_initializer = self.initializer
                ))
            
    @tf.Module.with_name_scope
    def __call__(self, x):
        x = 2.0*(x - self.lb)/(self.ub - self.lb) - 1.0
        for layer in self.layers:
            x = layer(x)
        return x
        
        
    def initialize_NN(self):
        """ Initialises a fully connected deep nerual network """
        model = keras.Sequential()
        model.add(keras.layers.Dense(self.layers[1] , input_shape=(self.layers[0],)))
        for ii in range(2, len(self.layers) - 2):
            model.add(keras.layers.Dense(units = self.layers[ii],
                            activation = self.activation,
                            kernel_initializer = self.initializer
                            ))
        model.add(keras.layers.Dense(units = self.layers[-1],
                activation = None,
                kernel_initializer = self.initializer
                ))
        return model
    
    def res_net_block(self, input_data, neurons):
      x = keras.layers.Dense(units = neurons, activation=self.activation,
                            kernel_initializer = self.initializer)(input_data)
      x = keras.layers.BatchNormalization()(x)
      x = keras.layers.Dense(units = neurons, activation=self.activation, 
                            kernel_initializer = self.initializer)(x) 
      x = keras.layers.BatchNormalization()(x)
      x = keras.layers.concatenate([x, input_data])
      x = keras.layers.Activation(self.activation)(x)
      return x
        
    def initialize_resnet(self):
        """ Initialises a Resnet """
        inputs = keras.Input(shape=self.num_inputs, )
        x = inputs 
        for ii in range(self.num_blocks):
            x = self.res_net_block(x, self.neurons)
        x = keras.layers.Dense(units = self.neurons, activation=self.activation, 
                            kernel_initializer = self.initializer)(x) 
        outputs = keras.layers.Dense(units = self.num_outputs, activation=None, 
                            kernel_initializer = self.initializer)(x) 
        model = keras.Model(inputs, outputs)
        return model
        
    
    def normalise(self, X):
        """ Performs Min-Max Normalisation on the input parameters using the predefined lower and uppwer ranges """
        return 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
    
    @tf.function
    @tf.autograph.experimental.do_not_convert
    def forward(self, model, X):
        """ Performs the Feedforward Operation """
        X_norm = self.normalise(X)
        return model(X_norm, training=True)
    
    