#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:58:15 2020

@author: Vicky

Neural PDE - Tensorflow 2.X
Module : PDE Module 

Module with the PDE class that takes in the user defined pde parameters and creates a symbolic expression which 
is executed using lambdify. 
"""

import numpy as np
import tensorflow as tf 
import sympy

from sympy.parsing.sympy_parser import parse_expr


class PDE(object):
    def __init__(self, eqn_str, in_vars, out_vars):
        """
        

        Parameters
        ----------
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
        self.num_inputs = len(in_vars)
        self.num_outputs = len(out_vars)
        
        in_vars = sympy.symbols(in_vars)
        out_vars = sympy.symbols(out_vars)
        
        if self.num_outputs ==1:
           self.all_vars = list(in_vars) + [out_vars]
        else: 
            self.all_vars = list(in_vars) + list(out_vars)

        self.expr = parse_expr(eqn_str)
    
        self.fn = sympy.lambdify(self.all_vars, self.expr, [{'D': self.first_deriv, 'D2': self.second_deriv, 'D3':self.third_deriv}, 'tensorflow'])
        
        

    def first_deriv(self, u, wrt):
        return tf.gradients(u, wrt)[0]
    

    def second_deriv(self, u, wrt):
        u_deriv = tf.gradients(u, wrt)[0]
        return tf.gradients(u_deriv, wrt)[0]
    

    def third_deriv(self, u, wrt):
        u_deriv = tf.gradients(u, wrt)[0]
        u_deriv = tf.gradients(u_deriv, wrt)[0]
        return tf.gradients(u_deriv, wrt)[0]
    
    @tf.function
    @tf.autograph.experimental.do_not_convert
    def func(self, X):
        
        t = X[:, 0:1]
        x = X[:, 1:2]
        u = self.model(tf.concat([t,x], 1), training=True)
        return self.fn(t, x, u) 

    

