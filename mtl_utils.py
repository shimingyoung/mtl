# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 12:28:02 2018

@authors: Rajat Patel, Shiming Yang
"""
#from numpy.linalg import norm

import numpy as np

def norm_11(a):
    m, n = a.shape
    for i in range(n):
        x = np.sum(a[:,i])
    y = np.sum(x)
    return(y)

def norm_1_infinity(a): # a is a 2-D array
    # return the sum of each row's maximum absolute values
    #print(a)
    y =[]
    m,n = a.shape
    for i in range(n):
        x = np.max(np.abs(a[:, i])) # is it absolute values?
        y.append(x)
    #print(y)
    y = np.sum(x)
    return(y)

def project_L1_ball(D, tau):
    # min (1/2) ||X-C||_2^2, s.t. ||X||_1 <= t
    return np.multiply(np.sign(D), np.max(0, abs(D)-tau) ) 

def proximal_L1_inf_norm(D, tau):
    # solve with L1inf constraint, solve its conjugate L_1
    # min_x ||x-D||_2^2 s.t. ||x||_1<=tau. then x = D - x
    # A. Quattoni,  et al. An Efficient Projection for l_{1,\inf} Regularization
    # use the eqs 15-19, and Lemma2
    
    m, n = np.shape(D)
    tol = 1e-8
    x = np.zeros((m,n))
    # calculate each row's abs sum
    s = np.sum(np.abs(D), axis = 0)
    for j in range(0, m): # for each j-th row
        
        # if ||D_j||_1 <= t, D_j is the solution
        if (s[j] <= tau):
            x[j, :] = D[j, :]
            continue
        # otherwise
        gnum = n
        theta = 0
        while (abs(s[j] - tau - gnum*theta) > tol):
            theta = (s - tau) / gnum
            s = 0
            rho = 0
            for i in range(0, gnum):
                if (x[i*m+j] >= theta):
                    x[rho*m+j] = x[i*m+j]
                    s[j] = s[j] + x[i*m+j]
                    rho = rho + 1
            gnum = rho
        # projection result
        # for this j-th row, for each i-th col
        x[j, :] = 0.0
        locs = D[j, :] > theta
        x[j, locs] = D[j, locs] - theta
        locs = D[j, :] < (-1*theta)
        x[j, locs] = D[j, locs] + theta

    return D - x

def gradient_log_loss(W, c, X, y):
    # W coef, d by 1, c 1 by 1, X n by d, y n by 1
    N = len(y)
    y_estimate = -y * (X.dot(W) + np.repeat(c, N))
    y_max = np.max(y_estimate, axis=0)
    weight_y = y / N
    func_val = (np.log(np.exp(-y_max) + np.exp(y_estimate - y_max) + y_max)) / N
    pp = 1 / (1 + np.exp(y_estimate))
    b = -weight_y * (1 - pp)
    grad_w = X * b
    grad_c = sum(b)
    return grad_w, grad_c, func_val
	
def dirty_model_logistic(n_task):
    #input X: n x d x t array, where n is number of observations, d is the number of variables, t is the number of tasks
    #	   Y: n x 1 x t array
    #	   lambda_s: 1 x 1
    #	   lambda_b: 1 x 1
    #output Theta: d x t array
    #	    B: d x t array
    #	    S: d x t array
    
    # A. Beck et al, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems
	
    # initialize
    B0 = np.zeros(d, t)
    S0 = np.zeros(d, t)
    obj_val = 0
    B = B0
    S = S0
    # reshape
    X = np.reshape(X, x*t, d)
    Y = np.reshape(Y, n_task*t, 1)
    # convert X to block diagonal matrix
    X = diagonalize(X)
    # Calculate 
    xtx = np.matmul(np.transpose(X), X)
    xty = np.transpose(X).dot(Y)
    Bn = B
    Sn = S
    L1norm = np.norm(X, ord='1')
    Linfnorm = np.norm(X, ord='inf')
    # calculate the upper bound of max eigenvalue of the hessian matrix
    L = 2 * min(L1norm*L1norm, n_task*t*Linfnorm^2, d*t*L1norm^2, n_task*d*t^2*max())
    obj_val = 0
	
    # iterationapple = web.DataReader("AAPL", "yahoo", start, end)
    for i in range(0, maxIter):
        B_old = B
        S_old = S
        t_old = t_new
        # calculate the gradient of log loss
        #grad_vec = 2 * (xtx * (np.respahe(Bn, -1, 'C') + np.reshape(Sn, -1, 'C')) - xty)
        grad_vec = gradient_log_loss(W, c, X, y)
        grad_mat = np.reshape(grad_vec, d, t)

        B = proximal_L1_inf_norm(Bn - gradmat / L, lambda_b / L)
        S = project_L1_ball(Sn - gradmat / L, lambda_s / L)
        obj_val = X.dot(B+S)
        # check termination condition

        # update with stepsize
        t_new = (1 + np.sqrt(1 + 4 * t_old^2)) / 2
        eta = (t_old - 1) / t_new
        Bn = B + eta * (B - B_old)
        Sn = S + eta * (S - S_old)
        
    return B, S

