#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:12:20 2020

@author: Vicky

Neural PDE - Tensorflow 2.X
Module : Plotting tools 
"""

import numpy as np
import matplotlib.pyplot as plt 


def evolution_plot(u_actual, u_pred):
    """
    Plots frame by frame evolution of the neural network solution in the 
    1D space along with the baseline solution

    Parameters
    ----------
    u_actual : 2D NUMPY ARRAY
        True solution 
    u_pred : 2D NUMPY ARRAY
        Neural Network Prediction

    Returns
    -------
    None.

    """
    
    actual_col = '#302387'
    nn_col = '#DF0054'
    plt.figure()
    plt.plot(0, 0, c = actual_col, label='Actual')
    plt.plot(0, 0, c = nn_col, label='NN', alpha = 0.5)
    plt.legend()
    for ii in range(len(u_actual)):
        plt.plot(u_actual[ii], c= actual_col, label = "Actual")
        plt.plot(u_pred[ii], c= nn_col, label = "NN")
        plt.legend()
        plt.pause(0.0001)
        plt.clf()

        
def space_time_error(u_actual, u_pred, x, t):
    """
    Plots the Absolute across the bounded space-time

    Parameters
    ----------
    u_actual : 2D NUMPY ARRAY
        True Solution
    u_pred : 2D Numpy Array
        Neural PDE Solution
    x : 1D NUMPY ARRAY
        x input array.
    t : 1D NUMPY ARRAY
        t input array.

    Returns
    -------
    None.

    """
    abs_error = np.abs(u_actual - u_pred)
    
    xx, tt = np.meshgrid(x, t)
    plt.figure()
    plt.tricontourf(xx.flatten(), tt.flatten(), abs_error.flatten(), cmap='plasma')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title("Absolute Error across Space Time")
    plt.colorbar()