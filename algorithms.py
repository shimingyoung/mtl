# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:43:23 2018

@author: Admin
"""

def fista_function(X, y):
    loss_old = 0------------------------------------
    lambda_1 = 1
    lambda_2 = 9
    L_0 = 15
    eta = 10
    random_number = 30
    S = np.random.normal(0, 1,(X_train.shape[1], y_train.shape[1]))
    B = np.random.normal(0,1, (X_train.shape[1], y_train.shape[1]))
    print(S.shape)
    print(B.shape)
    total_loss_list = []
    activation = np.zeros((y_train.shape[0], y_train.shape[1]))
    activation_1 = np.zeros((y_train.shape[0], y_train.shape[1]))
    max_iteration = 100
    precision = 1e-1
    p =0
    for j in range(max_iteration):
        for p in range(0,len(X_train)):
            hypothesis = np.dot(np.transpose(S+B), X_train[p])
            activation[p]= sigmoid_activation(hypothesis)
        loss_1 = log_loss(y_train, activation)
        print("This is first regularizer", norm_11(S))
        print("This is second regularizer", norm_1_infinity(B))
        total_loss = loss_1 + lambda_1*norm_11(S)+ lambda_2*norm_1_infinity(B)
        total_loss_list.append(total_loss)
        print(total_loss)
        if abs(loss_old - total_loss) == precision:
            return(S, B, total_loss_list, p, activation)
        gradient = gradient_logistic_function(activation, X_train, y_train)
        for i in range(random_number):
            if i == 0:
                L_new = (eta**i)*(L_0)
                print(L_new)
                t = 1
            else:
                L_new = (eta**i)*L_old
                print(L_new)
                t = 1/L_new
            proj_S = project_L1_ball((S - t*gradient).T, lambda_1)
            proj_B = projected_L1_infinity_ball((B - t*gradient).T, lambda_2)
            proj_S = proj_S.T
            proj_B = proj_B.T
            loss_old,log_loss_2,gradient_old = first_condition(proj_S, proj_B, X_train,y_train, lambda_1, lambda_2)
            b = second_order_expansion(log_loss_2, proj_S, proj_B, S, B, gradient_old, L_new, lambda_1, lambda_2)
            print("This is loss_old", loss_old)
            print("This is second_part", b)
            if loss_old <= b:
                S = proj_S - S
                B = proj_B - B
                break
            else:
                L_old = L_new
                S = proj_S - S
                B = proj_B - B             
#                if L_new == L_old:
#                    break
        print("This is the new L", L_new)
        t_1 = 1/L_new
        t_new = (1 + np.sqrt(1 + 4 *(t_1**2)))/2
        eta_1= (t_1-1)/t_new
        if j == 0:
            S_old = 0
            B_old = 0 
        else:
            S = S + eta_1*(S - S_old)
            B = B + eta_1*(B - B_old) 
        S_old = S
        B_old = B
        loss_old = total_loss
    return(S, B, total_loss_list, p, activation)
S, B, total_loss_list, p, activation_1 = fista_function(X_train, y_train)
print(p)
plt.plot(np.array(total_loss_list))
plt.show()


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
    
#    bias = np.random.normal(0,1,(X_train.shape[0],1))
#    bias = np.ones((X_train.shape[0],1))
#    for j in range(max_iteration):
#        loss_1 = 0
#        for p in range(len(X_train)):
#            hypothesis = np.dot(np.transpose(theta_1), X_train[p])
#            activation[p] = hypothesis
#        loss_1 = np.linalg.norm(np.subtract(y_train, activation))**2
#        total_loss = ((1/2) *loss_1 + lambda_1*(norm_11(S))+ lambda_2*norm_1_infinity(B))
#        print(total_loss)
#        total_loss_list.append(total_loss)
#        if abs(total_loss - loss_old) <= precision:
#            return(theta_1, S, B, total_loss_list, activation)
#        theta_1 = solvin_argmin_least_square(activation, X_train, y_train, theta_1, Z_k, U_k, rho)
#        proj_S = project_L1_ball((theta_1 -B + U_k), lambda_1/rho)
#        proj_B = projected_L1_infinity_ball((theta_1 -  S + U_k), lambda_2/rho)
#        Z_k = proj_S+ proj_B
#        U_k = U_k + theta_1 - Z_k
#        S = proj_S 
#        B = proj_B
#        loss_old = total_loss
#    print(j)
#    return(theta_1, S, B, total_loss_list, activation)
theta_1, S, B, total_loss_list, activation = ADMM_with_least_square(X_train, y_train)
plt.plot(np.array(total_loss_list))
plt.show()
#%%
w1 = roc_auc_score_1(y_train, activation_1)
print(w1) 