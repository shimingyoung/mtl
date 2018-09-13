# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:43:23 2018

@author: Admin
"""
from projections import project_L1_ball
from projections import project_L1_infinity_ball
from norms import norm_1_infinity
from norms import norm_11
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(Y2):
    return(1/(1+np.exp(-Y2)))

def logistic_loss_function(X1, Y1, S1, B1, lambda_1, lambda_2, n_var, n_task):
    N = len(X1)
    Y2 = np.matmul(X1,(B1+S1))
    Y2 = sigmoid(Y2)
    class_cost_1 = np.multiply(-Y1, np.log(Y2))
    class_cost_2 = np.multiply((1 -Y1), np.log(1 - Y2))
    total = class_cost_1 - class_cost_2
    total_loss = total.sum()/N
    final_loss = total_loss + lambda_1*norm_11(S1.reshape((n_var, n_task))) + lambda_2*norm_1_infinity(B1.reshape((n_var, n_task)))
    a1 = np.subtract(Y2, Y1)
    gradient = np.matmul(X1.T,a1)/N
    return(final_loss, total_loss, gradient)

def first_condition(proj_S, proj_B, X1, Y1, lambda_1, lambda_2, loss_option, n_var, n_task):
    s1 = proj_S.ravel().reshape(-1,1)
    b1 = proj_B.ravel().reshape(-1,1)
    if loss_option == "logistic":
        loss, log_loss, gradient_updated = logistic_loss_function(X1, Y1, s1, b1, lambda_1, lambda_2, n_var, n_task )
        return(loss, log_loss, gradient_updated)
    else:
        activation = np.matmul(X1,(s1+b1))
        log_loss_2 = (np.linalg.norm(Y1 - activation))**2
        loss =  log_loss_2 + lambda_1*norm_11(proj_S) + lambda_2*norm_1_infinity(proj_B)
        a1 = np.subtract(activation, Y1)
        gradient_updated = 2*np.matmul(X1.T,a1)
        gradient_updated = gradient_updated.reshape(proj_S.shape[0], proj_S.shape[1])
        return(loss,log_loss_2, gradient_updated)

def second_order_expansion(loss_old, proj_S, proj_B, S, B, gradient_old, L, lambda_1, lambda_2):
    first_term = loss_old
    S = S.reshape(proj_S.shape[0], proj_S.shape[1])
    B = B.reshape(proj_B.shape[0], proj_B.shape[1])
    x_k = (proj_S+proj_B - (S+B))
    second_term = np.inner((x_k).ravel(), gradient_old.ravel())    
    third_term = L/2*((np.linalg.norm((x_k), ord=2))**2)
    fourth_term = lambda_1*norm_11(proj_S)+ lambda_2*norm_1_infinity(proj_B)
    output = first_term+second_term+third_term+fourth_term
    return(output)

def FISTA_algorithm_multitasking(X1, Y1,  n_var, n_task, max_iteration, loss_option=None):
    n_obs= X1.shape[0]
#    n_task = 139
#    n_var = 28
    plt.figure(figsize=(1,1))
    lambda_1 = 1
    lambda_2 = 1
    eta = 3
    S1 = np.random.normal(0,1, (n_var, n_task))
    B1 = np.random.normal(0,1, (n_var, n_task))
    S1 = S1.flatten()
    B1 = B1.flatten()
    S_old = S1
    B_old = B1
    final_loss_list = []
    Y2 = np.zeros((Y1.shape[0], Y1.shape[1]))
#    max_iteration = 2000
    loss_old = 0
    rand_iteration = 50
    L_new = 10
    precision = 1e-3
    t_1 = 1
    total_loss_list = []
    norm1_list = []
    norminf_list = []
    for i in range(max_iteration):
        if loss_option == "logistic":
            final_loss, total_loss, gradient = logistic_loss_function(X1, Y1, S1, B1, lambda_1, lambda_2, n_var, n_task)
        else:
            Y2 = np.matmul(X1, (B1+S1))
            Y2 = Y2.reshape(-1,1)
            total_loss = np.linalg.norm(Y2- Y1)**2
            total_loss_list.append(total_loss)
            final_loss = total_loss + lambda_1*norm_11(S1.reshape((n_var, n_task))) + lambda_2*norm_1_infinity(B1.reshape((n_var, n_task)))
            print(final_loss)
            final_loss_list.append(final_loss)
            a1 = np.subtract(Y2, Y1)
            gradient = 2*np.matmul(X1.T,a1)
        if abs(loss_old - final_loss) <= precision:
            return(S1, B1)
        norm1_list.append(norm_11(S1.reshape((n_var, n_task))))
        norminf_list.append(norm_1_infinity(B1.reshape((n_var, n_task))))
        """
        Now trying to find best possible Lipchitz constant for the given function
        """
        for j in range(rand_iteration):
            L_new = (eta**j)*(L_new)
            proj_S = project_L1_ball((S1.reshape((n_var, n_task)) - gradient.reshape((n_var, n_task))/L_new), lambda_1/L_new)        
            proj_B = project_L1_infinity_ball((B1.reshape((n_var, n_task)) - gradient.reshape((n_var, n_task))/L_new), lambda_2/L_new)
            """ checking the condition"""
            loss_old,log_loss_2,gradient_old = first_condition(proj_S, proj_B, X1 ,Y1, lambda_1, lambda_2, loss_option, n_var, n_task)
            b = second_order_expansion(total_loss, proj_S, proj_B, S1, B1, gradient, L_new, lambda_1, lambda_2)
            if loss_old <= b:
                break
            else:
                print("Condition not true")
        """Now since we have found the Lipchitz constant we update the parameters"""
        proj_S = project_L1_ball((S1.reshape((n_var, n_task)) - gradient.reshape((n_var, n_task))/L_new), lambda_1/L_new)
        proj_B = project_L1_infinity_ball((B1.reshape((n_var, n_task)) - gradient.reshape((n_var, n_task))/L_new), lambda_2/L_new)
        S1 = proj_S.ravel()
        B1 = proj_B.ravel()
        t_new = (1 + np.sqrt(1 + 4 *(t_1**2))/2)
        eta_1= (t_1-1)/t_new
        S1 = S1 + eta_1*(S1 - S_old)    
        B1 = B1 + eta_1*(B1 - B_old)
#        diff = np.linalg.norm(S1)
        S_old = S1
        B_old = B1
        loss_old = final_loss
        t_1 = t_new
#    plt.matshow(S1.reshape((n_var, n_task)))
#    plt.matshow(B1.reshape((n_var, n_task)))
    return(S1, B1)
#%%
def ADMM_with_least_square(X_train, y_train):
    loss_old = 0
    lambda_1 = 0.009
    lambda_2 = 0.009
    U_k = np.random.normal(0, 1,(X_train.shape[1], y_train.shape[1]))
    rho = 0.008
    S = np.random.normal(0, 1,(X_train.shape[1], y_train.shape[1]))
    B = np.random.normal(0,1, (X_train.shape[1], y_train.shape[1]))
    total_loss_list = []
    activation = np.zeros((y_train.shape[0], y_train.shape[1]))
#    activation_1 = np.zeros((y_train.shape[0], y_train.shape[1]))
    max_iteration = 150
    precision = 1e-8
    theta_1 = S+B
    Z_k = S+B
    N = len(X_train)
    bias = np.random.normal(0,1,(X_train.shape[0],1))
    bias = np.ones((X_train.shape[0],1))
    for j in range(max_iteration):
        loss_1 = 0
        for p in range(len(X_train)):
            hypothesis = np.dot(np.transpose(theta_1), X_train[p])
            activation[p] = hypothesis
        loss_1 = np.linalg.norm(np.subtract(y_train, activation))**2
        total_loss = ((1/2) *loss_1 + lambda_1*(norm_11(S))+ lambda_2*norm_1_infinity(B))
        print(total_loss)
        total_loss_list.append(total_loss)
        if abs(total_loss - loss_old) <= precision:
            return(theta_1, S, B, total_loss_list, activation)
        theta_1 = solvin_argmin_least_square(activation, X_train, y_train, theta_1, Z_k, U_k, rho)
        proj_S = project_L1_ball((theta_1 -B + U_k), lambda_1/rho)
        proj_B = projected_L1_infinity_ball((theta_1 -  S + U_k), lambda_2/rho)
        Z_k = proj_S+ proj_B
        U_k = U_k + theta_1 - Z_k
        S = proj_S 
        B = proj_B
        loss_old = total_loss
    print(j)
    return(theta_1, S, B, total_loss_list, activation)
#theta_1, S, B, total_loss_list, activation = ADMM_with_least_square(X_train, y_train)
#plt.plot(np.array(total_loss_list))
#plt.show()
#%%

#def fista_function(X, y):
#    loss_old = 0------------------------------------
#    lambda_1 = 1
#    lambda_2 = 9
#    L_0 = 15
#    eta = 10
#    random_number = 30
#    S = np.random.normal(0, 1,(X_train.shape[1], y_train.shape[1]))
#    B = np.random.normal(0,1, (X_train.shape[1], y_train.shape[1]))
#    print(S.shape)
#    print(B.shape)
#    total_loss_list = []
#    activation = np.zeros((y_train.shape[0], y_train.shape[1]))
#    activation_1 = np.zeros((y_train.shape[0], y_train.shape[1]))
#    max_iteration = 100
#    precision = 1e-1
#    p =0
#    for j in range(max_iteration):
#        for p in range(0,len(X_train)):
#            hypothesis = np.dot(np.transpose(S+B), X_train[p])
#            activation[p]= sigmoid_activation(hypothesis)
#        loss_1 = log_loss(y_train, activation)
#        print("This is first regularizer", norm_11(S))
#        print("This is second regularizer", norm_1_infinity(B))
#        total_loss = loss_1 + lambda_1*norm_11(S)+ lambda_2*norm_1_infinity(B)
#        total_loss_list.append(total_loss)
#        print(total_loss)
#        if abs(loss_old - total_loss) == precision:
#            return(S, B, total_loss_list, p, activation)
#        gradient = gradient_logistic_function(activation, X_train, y_train)
#        for i in range(random_number):
#            if i == 0:
#                L_new = (eta**i)*(L_0)
#                print(L_new)
#                t = 1
#            else:
#                L_new = (eta**i)*L_old
#                print(L_new)
#                t = 1/L_new
#            proj_S = project_L1_ball((S - t*gradient).T, lambda_1)
#            proj_B = projected_L1_infinity_ball((B - t*gradient).T, lambda_2)
#            proj_S = proj_S.T
#            proj_B = proj_B.T
#            loss_old,log_loss_2,gradient_old = first_condition(proj_S, proj_B, X_train,y_train, lambda_1, lambda_2)
#            b = second_order_expansion(log_loss_2, proj_S, proj_B, S, B, gradient_old, L_new, lambda_1, lambda_2)
#            print("This is loss_old", loss_old)
#            print("This is second_part", b)
#            if loss_old <= b:
#                S = proj_S - S
#                B = proj_B - B
#                break
#            else:
#                L_old = L_new
#                S = proj_S - S
#                B = proj_B - B             
##                if L_new == L_old:
##                    break
#        print("This is the new L", L_new)
#        t_1 = 1/L_new
#        t_new = (1 + np.sqrt(1 + 4 *(t_1**2)))/2
#        eta_1= (t_1-1)/t_new
#        if j == 0:
#            S_old = 0
#            B_old = 0 
#        else:
#            S = S + eta_1*(S - S_old)
#            B = B + eta_1*(B - B_old) 
#        S_old = S
#        B_old = B
#        loss_old = total_loss
#    return(S, B, total_loss_list, p, activation)
#S, B, total_loss_list, p, activation_1 = fista_function(X_train, y_train)
#print(p)
#plt.plot(np.array(total_loss_list))
#plt.show()



#%%