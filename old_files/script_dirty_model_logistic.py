# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:31:59 2018

@author:
"""

# this is a script to run the dirty model
#sys.path.append(os.path.relpath(''))
import mtl_utils

# Read data

# partition dataset for CV

# prepare X and Y. For dirty models, d is the same among all tasks
# input X: a tuple of (n x d) arrays, Y: n x 1 x t array

B, S, C = mtl_utils.dirty_model_logistic(X, Y, maxIter = 100)
W = B + S

# evaluate model on the testing set

# save the model and its performance on this dataset