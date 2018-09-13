# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:37:32 2018

@author: Admin
"""
import numpy as np
from norms import *

# a collection of projection operators
def project_L1_ball(D, tau):
    # min (1/2) ||X-C||_2^2, s.t. ||X||_1 <= t
    if norm_11(D) <= tau:
        return(D)
    else:
        q = np.sign(D)
    #    print(q)
        a = np.zeros((D.shape[0], D.shape[1]))
        for i in range(len(D)):
            for j in range(D.shape[1]):
                a[i][j] = np.maximum(0, (abs(D[i][j])-tau))
        return np.multiply(a,q)    

def project_L1_infinity_ball(A, C):
    C = C*norm_1_infinity(A)
    if norm_1_infinity(A) <= C:
        return(A)
    else:
        Q = float(norm_1_infinity(A))
        W = np.absolute(A)
        #print(W)
        C1 = 0.001*Q
        B = np.zeros((A.shape[0], A.shape[1]))
        S = np.zeros((W.shape[0], W.shape[1]))
        for i in range(W.shape[0]):
            S[i] = np.sort(W[i], kind="mergesort")
            S[i] = S[i][::-1]
        y = np.array(np.repeat([0], S.shape[0]),)
        y = y.reshape(1, -1)
        S = np.append(S, y.T, axis=1)
        S1 = S.ravel()
        r = np.zeros((S.shape[0], S.shape[1]))
        row_index_list = []
        for j in range(S.shape[0]):
            for k in range(S.shape[1]):
                r[j][k] = (np.sum(S[j][:k] - S[j][k]))
                row_index_list.append(j)
        #        print(r[j][k])
        r1 = r.ravel()
        r2_array = np.argsort(r1)
        r1 = r1[r2_array]
        ks = np.zeros((len(W),))
        idx = 0
        while r1[idx] <= 0:
            a1 = row_index_list[r2_array[idx]]
            ks[a1] = ks[a1]+1
            idx = idx+1
        gradient = 0
        for i in range(len(ks)):
            gradient += -1/ks[i]
        norm = Q
        idx = idx-1
        next_norm = gradient*(r1[idx+1] - r1[idx]) + norm
        while (next_norm > C1):
            idx = idx+1
            norm = next_norm
            r_1 = row_index_list[r2_array[idx]]
            if True:
                gradient+=(-1/ks[r_1])
                ks[r_1] = ks[r_1] +1
                if ks[r_1] <= W.shape[1]:
                    gradient += 1/ks[r_1]
            else:
                ks[r_1] = ks[r_1]+1
                gradient  = 0
                for p in range(len(ks)):
                    if ks[p] <= W.shape[1]:
                        gradient += -1/ks[p]
            next_norm = gradient *(r1[idx+1] - r1[idx]) + norm
        theta = (C - norm + gradient*r1[idx])/gradient
        #calculating mu
        U = np.zeros((len(W), ))
        sum_1 = 0
        for x in range(S.shape[0]):
            if ks[x] <= A.shape[1]:
                y = idx
                while (row_index_list[r2_array[y]] != x):
                    y = y-1
                    U[x] = S1[int(x*(S.shape[1])+ks[x]-1)] - (1/ks[x])*(theta - r1[y])
            else:
                U[x] = 0    
            sum_1+= U[x]
        for c in range(A.shape[0]):
            for d in range(A.shape[1]):
                if A[c][d] >= U[c]:
                    B[c][d] = U[c]
                elif A[c][d] <= U[c]:
                    B[c][d] = A[c][d]
                else:
                    B[c][d] = 0
        return(np.sign(A)*B)    


def project_L1inf(D, tau):
    B = np.copy(D) # make a hard copy of D
    
    l1infnorm_D = norm_1_infinity(D)
    if l1infnorm_D <= tau:
        return B
    else:
        nRow, nCol = np.shape(D)
        rowAbsSum = np.sum(np.fabs(D), axis = 1)
        nonZeroRows = np.where(rowAbsSum > 0)
        
        S = np.zeros(nonZeroRows, nCol+1)
        R = np.zeros(nonZeroRows, nCol+1)
        R_idx = np.zeros(nonZeroRows, nCol+1)
        
        for i in nonZeroRows:
            S[i, 0:-1] = np.sort(np.fabs(A[i,]))[::-1]
        
        for i in nonZeroRows:
            for j in nCols:
                R_idx[i,j]  = i
                R[i, j+1] =  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        