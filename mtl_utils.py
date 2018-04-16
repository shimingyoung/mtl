# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 12:28:02 2018

@authors: Rajat Patel, Shiming Yang
"""
from numpy.linalg import norm
from numpy.linalg import svd
from scipy.sparse import block_diag

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

def proximal_trace_norm(D, tau):
    # trace norm, or for low rank regulation
    # Liu, Ji, Ye 2009, Multi-Task Feature Learning Via Efficient L_{2,1}-Norm Minimization
    m, n= np.shape(D)
    if (m > n):
        U, S, V = svd(D)
    else:
        U, S, V = svd(np.transpose(D))
        
    thres = np.diagonal(S) - tau * 0.5
    diag_S = thres * (thres > 0)
    D_p = np.matmul(np.matmul(U, np.diagflat(diag_S)), np.transpose(V))
    trace_D = sum(diag_S)
    return D_p, trace_D

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
	
def dirty_model_logistic(X, Y, lambda_b, lambda_s, maxIter = 200, opt=opt):
    #input X: tuple of n x d arrays, with n_task elements.
    #opt is a dict that stores the parameters and their default values
    #where n is number of observations, d is the number of variables, n_task is the number of tasks
    #	   Y: n x 1 x t array
    #	   lambda_s: 1 x 1
    #	   lambda_b: 1 x 1
    #output
    #	    B: d x t array
    #	    S: d x t array
    #      c: 1 x t array
    
    # Solvers (choose one):
    # FISTA: A. Beck et al, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems
    # ADMM: S. Boyd, Proximal Algorithms (p155); or S. Boyd, Distributed Optimization and Statistical Learning via ADMM
    # Fast ADMM: T. Goldstein et al, Fast Alternating Direction Optimization Methods
    n, d, n_task = X.shape
    # initialize
    B0 = np.zeros(d, n_task)
    S0 = np.zeros(d, n_task)
    c = np.zeros(d, 1)
    obj_val = 0
    obj_val_old = 0
    tol = 1e-8
    B = B0
    S = S0
    # reshape
    #X = np.reshape(X, n*n_task, d)
    Y = np.reshape(Y, n*n_task, 1)
    # convert X to block diagonal matrix
    #X = diagonalize(X)
    X = block_diag(X) # block diagonalize X in sparse
    # Calculate 
    #xtx = np.matmul(np.transpose(X), X)
    #xty = np.transpose(X).dot(Y)
    Bn = B
    Sn = S
    L1norm = norm(X, ord='1')
    Linfnorm = norm(X, ord='inf')
    # calculate the upper bound of max eigenvalue of the hessian matrix
    # refer to C.L. Byrne, 'Iterative Optimization in Inverse Problems', Section 9.5
    # also review the Taylor expansion, function f(x) near the point x_k, can be approximated by
    # f(x) \approx= f(x_k) + f'(x_k) * (x-x_k) + f''(x_k) * (x - x_k)^2 / 2! + ...
    # in a inner product way, it can be written as
    # f(x) \approx= f(x_k) + <f'(x_k), x-x_k> + L/2! * ||x-x_k||_2^2
    # the f''(x_k), a.k.a. the hessian matrix, is difficult to calculate in high dimension. So need to find an easy estimation,
    # e.g its upper bound
    L = 2 * min(L1norm*L1norm, n_task*n*Linfnorm^2, d*n_task*L1norm^2, n_task*n_task*d*n*np.amax(abs(X)))
    t_new = 0
	
    if opt.solver == 'fista':
        # FISTA
        for i in range(0, maxIter):
            B_old = B
            S_old = S
            t_old = t_new
            obj_val_old = obj_val
            # calculate the gradient of log loss
            #grad_vec = 2 * (xtx * (np.respahe(Bn, -1, 'C') + np.reshape(Sn, -1, 'C')) - xty)
            grad_vec, grad_c, obj_val = gradient_log_loss(B+S, c, X, Y)
            grad_mat = np.reshape(grad_vec, d, n)
            
            # check termination condition
            if (i>=5 and abs(obj_val - obj_val_old) <= tol) or i >= maxIter:
                break
    
            B = proximal_L1_inf_norm(Bn - grad_mat / L, lambda_b / L)
            S = project_L1_ball(Sn - grad_mat / L, lambda_s / L)
            c = c - grad_c / L
            #obj_val = X.dot(B+S)
            
            # update with stepsize
            t_new = (1 + np.sqrt(1 + 4 * t_old^2)) / 2
            eta = (t_old - 1) / t_new
            Bn = B + eta * (B - B_old)
            Sn = S + eta * (S - S_old)
            
            
        return B, S, c
    elif opt.solver == 'admm':
        for i in range(0, maxIter):
            # TODO
            
        return B, S, c
    elif opt.solver == 'fadmm':
        for i in range(0, maxIter):
            #TODO
            
        return B, S, c
    else:
        print("Not a supported solver name.")
        # raise error here.
        

