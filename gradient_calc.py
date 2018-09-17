#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 14:53:58 2018

@author: shiming
"""

import numpy as np

def logit_gradient(w, c, x, y):
# find gradients for parameters w and c, for one task of logistic regression
   n_obs = len(y)
   weight = np.ones((n_obs,1)) / float(n_obs)
   tmp1 = -y * (np.multiply(np.transpose(x), w) + c)
   tmp2 = np.clip(tmp1, 0, None)
   tmp3 = np.log( np.exp(-tmp2) + np.exp(tmp1-tmp2) ) + tmp2
   funVal = np.dot(weight, tmp3)
   tmp4 = 1.0 / (1.0 + np.exp(tmp1))
   tmp5 =  np.multiply( -1.0 * np.multiply(weight, y), (1.0-tmp4))
   grad_w = np.dot(x, tmp5)
   grad_c = np.sum(tmp5)
   return grad_w, grad_c, funVal

def logit_funValue(w, c, x, y):
   n_obs = len(y)
   weight = np.ones((n_obs,1)) / float(n_obs)
   tmp1 = -y * (np.multiply(np.transpose(x), w) + c)
   tmp2 = np.clip(tmp1, 0, None)
   tmp3 = np.log( np.exp(-tmp2) + np.exp(tmp1-tmp2) ) + tmp2
   funVal = np.dot(weight, tmp3)
   return funVal

def logit_gradient_group(W, C, X, Y):
   n_var, n_task = W.shape
   grad_W = np.zeros(n_var, n_task)
   grad_C = np.zeros(1, n_task)
   funVals = np.zeros(1, n_task)
   
   for i in range(n_task):
       grad_W[:,i], grad_C[:,i], funVals[:,i] = logit_gradient(W[:,i], C[i], X[i], Y[i])
       
   return grad_W, grad_C, funVals

def logit_funValue_group(W, C, X, Y):
    funVal = 0
    n_var, n_task = W.shape
    for i in range(n_task):
        funVal = funVal + logit_funValue(W[:,i], C[i], X[i], Y[i])
    return funVal
