# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:37:32 2018

@author: S. Yang
"""
import numpy as np
#from norms import norm_1_infinity
import copy

#from norms import norm_11
# a collection of projection operators
def project_L11(D, tau):
    # min (1/2) ||X-C||_2^2, s.t. ||X||_1 <= t
    X = np.multiply(np.sign(D), np.clip(np.absolute(D)-tau, 0, None) )
    return X

def project_L1inf(A, C):
    # follow the c code 'projL1Inf.c' from https://github.com/doksa/altginv/blob/master/projL1Inf.c
    maxes = np.max(np.absolute(A), axis = 1)
    normOfA = np.sum(maxes)
    if normOfA <= C:
        return A
    else:
        #indices where maxes > 0
        nRows, nCols = A.shape
        sparseMap = np.where(maxes > 0)
        n_nonzeros = len(sparseMap[0]) # number of rows that row abs sum >0

        S = np.zeros([n_nonzeros, nCols+1])
        S[:,0:nCols] = np.absolute(A)
        
        sortedS = -np.sort(-S, axis = 1, kind='quicksort')
        res = np.cumsum(sortedS, axis = 1)
        J = np.tile(range(1,(nCols+1)), (n_nonzeros, 1) )
        Rvalue = np.hstack((np.zeros([n_nonzeros,1]), res[:,0:nCols] - np.multiply(sortedS[:,1:(nCols+1)], J) ) )
        RrowInd = np.transpose(np.tile(range(0,n_nonzeros), (nCols+1,1)))
        R = Rvalue.ravel(order='F')
        Rind = RrowInd.ravel(order='F')
        
        locs = np.argsort(R, kind='quicksort')
        R = R[locs]
        Rind = Rind[locs]

        idxR = 0
        Ks = np.zeros([n_nonzeros,1])
        
        while (R[idxR] <= 0.0):
            Ks[Rind[idxR]] = Ks[Rind[idxR]] + 1
            idxR = idxR + 1
        idxR = idxR - 1
        
        Gradient = sum( -1.0 / Ks)
        norm = normOfA
        nextNorm = Gradient * (R[idxR+1] - R[idxR]) + norm
        
        while (nextNorm > C):
            idxR = idxR + 1
            norm = nextNorm
            r1 = Rind[idxR]
            Gradient = Gradient + 1.0 / Ks[r1]
            Ks[r1] = Ks[r1] + 1
            if (Ks[r1] <= nCols):
                Gradient = Gradient - 1.0/Ks[r1]
            nextNorm = Gradient * (R[idxR+1] - R[idxR]) + norm
            
        theta = (C - norm + Gradient * R[idxR]) / Gradient
        
        #sum1 = 0
        mu = np.zeros([n_nonzeros, 1])
        for i in range(n_nonzeros):
            if Ks[i] <= nCols:
                j = idxR
                while Rind[j] != i:
                    j = j - 1
                mu[i] = sortedS[i][int(Ks[i]-1.0)] - (theta - R[j]) / Ks[i]
            else:
                mu[i] = 0.0
            #sum1 = sum1 + mu[i]
        
        B = copy.deepcopy(A)
        for i in range(n_nonzeros):
            flag = np.logical_or(B[i,:] > mu[i], B[i,:] < -mu[i])
            loc1 = np.where(np.logical_and(flag, A[i,:] >= 0.0))
            B[i, loc1] = mu[i]
            loc1 = np.where(np.logical_and(flag, A[i,:] < 0.0))
            B[i, loc1] = -mu[i]
    return B


def project_L21(D, tau):
    n_var, n_task = D.shape
    c = np.clip(1 - tau / np.sqrt(np.sum(np.square(D), axis=1) ), 0, None)
    X = np.multiply(np.tile(c, [1, n_task]), D)
    return X
    
def project_trace(D, tau):
    n_var, n_task = D.shape
    if (n_var > n_task):
        U, S, V = np.linalg.svd(D, full_matrices=True)
    else:
        U, S, V = np.linalg.svd(np.transpose(D), full_matrices=True)
    thres = np.diag(S) - tau / 2.0
    diag_S = np.clip(np.diag(S) - tau / 2.0, 0.0, None)
    L_hat = U * np.diag(thres) * V
    L_tn = sum(diag_S)
    return L_hat, L_tn

def project_box(v, l, u):
    #Project a point onto a box (hyper-rectangle)
    x = np.clip(v, l, u)
    return x