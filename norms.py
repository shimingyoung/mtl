# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:28:41 2018

@author: Admin
"""
# 

import numpy as np

def norm_11(X):
    m, n = X.shape
    norm = 0
    for i in range(m):
        norm += np.sum(np.absolute(X[i]))
    return(norm)

def norm_1_infinity(X):
    y =[]
    m,n = X.shape
    for i in range(m):
        row_inf_norm = np.max(np.absolute(X[i]))
        y.append(row_inf_norm)
    y = np.sum(y)
    return(y)
    
