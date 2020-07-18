#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:34:39 2020

@author: Vicky

Neural PDE - Tensorflow 2.X

Module: Quasi-Newtonian Wrappers 

Help convert the Model variables 1D and back to the Model structure. 
"""
import tensorflow as tf
import numpy as np
import time 


def TFP_Keras_Wrapper(model, loss_func, X_i, u_i, X_b, u_b, X_f):
     #Borrowed from Py Chao https://pychao.com/2019/11/02/optimize-tensorflow-keras-models-with-l-bfgs-from-tensorflow-probability/
    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)
    
    loss = []

    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))
            
            
    def val_and_grads_1d(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`
        """
        
        start_time = time.time()

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = loss_func(X_i, u_i, X_b, u_b, X_f)
        

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        val_and_grads_1d.iter.assign_add(1)       
        print('QN.  It: %d, Loss: %.3e, Time: %.2f' % 
                (val_and_grads_1d.iter, loss_value,  np.round(time.time() - start_time, 3)))
        # tf.print("Iter:", val_and_grads_1d.iter, "Loss:", loss_value, "Time:", np.round(time.time() - start_time, 2))

        return loss_value, grads
    
        # store these information as members so we can use them outside the scope
    val_and_grads_1d.iter = tf.Variable(0)
    val_and_grads_1d.idx = idx
    val_and_grads_1d.part = part
    val_and_grads_1d.shapes = shapes
    val_and_grads_1d.assign_new_model_parameters = assign_new_model_parameters
    val_and_grads_1d.loss = loss

    return val_and_grads_1d

def Scipy_Keras_Wrapper(model, loss_func, X_i, u_i, X_b, u_b, X_f):
     #Borrowed from Py Chao https://pychao.com/2019/11/02/optimize-tensorflow-keras-models-with-l-bfgs-from-tensorflow-probability/
    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices
    
    loss = []

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))
            
            
    def val_and_grads_1d(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """
        
        start_time = time.time()

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = loss_func(X_i, u_i, X_b, u_b, X_f)
        

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        val_and_grads_1d.iter.assign_add(1)
        print('QN.  It: %d, Loss: %.3e, Time: %.2f' % 
                (val_and_grads_1d.iter, loss_value,  np.round(time.time() - start_time, 3)))
        # tf.print("Iter:", val_and_grads_1d.iter, "Loss:", np.round(loss_value, 5), "Time:", np.round(time.time() - start_time, 2))

        return loss_value.numpy(), grads.numpy()
    
        # store these information as members so we can use them outside the scope
    val_and_grads_1d.iter = tf.Variable(0)
    val_and_grads_1d.idx = idx
    val_and_grads_1d.part = part
    val_and_grads_1d.shapes = shapes
    val_and_grads_1d.assign_new_model_parameters = assign_new_model_parameters
    val_and_grads_1d.loss = loss

    return val_and_grads_1d
