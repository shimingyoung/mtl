# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 15:45:00 2018

@author: rajat.patel
"""
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import MultiTaskLassoCV
import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from scipy import linalg
from sklearn.metrics import log_loss
# Reading of the input data file
input_data = pd.read_csv("C:\MTL_data\onpoint5_outcomes_features_02102018.csv")


# print(input_data.head)
# print(len(input_data))
# print(input_data.describe())

# print(input_data.columns)

#Reading the header files
def header(dataframe):
    header = []
    for i in input_data.columns:
        header.append(i)
    return header


def preprocessing_data(dataframe):
    count_null = dataframe.isnull().sum()
    return (count_null)


def test_label(dataframe):
    # finding header for the columns in input file
    column_label = header(dataframe)
    test_label_1 = dataframe[["MT", "CAT"]]
    test_label = dataframe[['MT', 'CAT', 'UnX', "mortality", "LOS>=3", "LOS>=7", "iss>=16", "iss>=25"]]
    dataframe.drop(['Accn', 'MT', 'CAT', 'UnX', "mortality", "LOS>=3", "LOS>=7", "iss>=16", "iss>=25"], axis=1,
                   inplace=True)
    return test_label, dataframe, test_label_1




def check_standard_deviation(feature_dataset):
    column_labels = header(feature_data)
    for i in column_labels:
        standard_deviation = np.std(feature_data[i])
        if standard_deviation < 0.2:
            feature_data.drop(i, axis=1, inplace=True)
    column_labels = header(feature_data)
    count_col = len(column_labels)
    return feature_data, count_col


def preprocessing_reduced_dataset(feature_dataset):
    column = header(feature_data)
    # Normalizing the values of all the columns
    count = 0
    column = header(feature_data)
    mean_collector = []
    for i in column:
        if feature_data[i].isnull().values.all():
            feature_data.drop(i, axis=1, inplace=True)
            count += 1
        elif feature_data[i].isnull().sum() >= 500:
            feature_data.drop(i, axis=1, inplace=True)
            count += 1
        else:
            mean_collector.append(np.mean(feature_data[i]))
    return max(mean_collector), min(mean_collector), len(mean_collector), count

def norm_11(a):
    m, n = a.shape
    for i in range(n):
        x = np.sum(a[:,i])
    y = np.sum(x)
    return(y)

def norm_1_infinity(a):
    #print(a)
    y =[]
    m,n = a.shape
    for i in range(n):
        x = np.max(a[:, i])
        y.append(x)
    #print(y)
    y = np.sum(x)
    return(y)

def proximal_norm_11(a, lambda1, t, gradient):
    matrix_norm = a - t*gradient
    m,n = matrix_norm.shape
    matrix_norm_1 = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if matrix_norm[i][j] > lambda1*t:
                matrix_norm_1[i][j] = matrix_norm[i][j] + lambda1*t
            else:
                matrix_norm_1[i][j] = matrix_norm_1[i][j] - lambda1*t
    return(matrix_norm_1)

def proximal_norm_1_infinity(a, lambda1, t, gradient):
    matrix_norm = a - t*gradient
    m,n = matrix_norm.shape
    matrix_norm_1 = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if matrix_norm[i][j] > lambda1*t:
                matrix_norm_1[i][j] = matrix_norm[i][j] + lambda1*t
            else:
                matrix_norm_1[i][j] = matrix_norm_1[i][j] - lambda1*t
    return(matrix_norm_1)

def sigmoid_activation(activation):
    activation = np.float_(activation)
    a = np.zeros((len(activation), 1))
    for i in range(len(activation)):
        a[i] = (1/(1+np.exp(-activation[i])))
        if a[i] >= 0.5:
            a[i] = 1
        else:
            a[i] = 0
    return(a)

def block_diagonal(a):
    X = linalg.block_diag(a)
    return(a)

#def logistic_regression_vectorize(y_label, activation,S, B, lambda_1, lambda_2, N):
#    loss = 1/N*((np.dot(-(y_label), np.log(activation))) - np.dot((1-y_label), np.log(1-activation)))
#    norms = lambda_1* norm_11(S) + lambda_2*norm_1_infinity(B)
#    final_loss = loss + norms
#    return(final_loss)
    
def gradiant_calculation(activations, X_train, y_train, N):
    gradient = np.dot(np.transpose(X_train), activations - y_train)
    return(gradient)

def logistic_regression_loss(y_label, activation, S, B, lambda_1, lambda_2, N):
    a = np.zeros((y_label.shape[0],))
    for i in range(len(y_label)):
        if y_label[i] == 0:
            a[i] = np.log(1 - activation[i])
        else:
            a[i] = np.log(activation[i])
    loss_part_1 = -1*np.sum(a)/N
    loss_part_2 = lambda_1*norm_11(S) + lambda_2*norm_1_infinity(B) 
    total_loss= loss_part_1+loss_part_2
    return(total_loss)
    
test_data, feature_data, test_label_1 = test_label(input_data)
feature_data, reduced_col_count = check_standard_deviation(feature_data)
feature_data.fillna(0, inplace=True)
#max1, min1, total_feature, null_count = preprocessing_reduced_dataset(feature_data)
#print(max1, min1, total_feature, null_count)
#print(reduced_col_count)
#print(feature_data.shape)
#print(test_data.shape)
#print(test_label_1.shape)
X = np.array(feature_data)

for i in range((X.shape[0])):
    for j in range((X.shape[1])):
        if not np.isfinite(X[i][j]):
            X[i][j] = 0
#print(X.shape)
X = block_diagonal(X)
y = np.array(test_data)
#print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 43)
lambda_1 = 0.0000001
lambda_2 = 0.0000002
#print(X_train.shape)
#print(X_test.shape)

#print(y_train.shape)
#print(y_test.shape)
alpha = 0.000000000001
S = np.random.normal(0, 1,(X_train.shape[1], y_train.shape[1]))
B = np.random.normal(0.5,0.002,(X_train.shape[1], y_train.shape[1]))
gradient_1 =  np.zeros((X_train.shape[1], y_train.shape[1]))
theta = np.zeros((X_train.shape[1], y_train.shape[1]))
bias = np.ones((y_train.shape[1],1))
activation = np.zeros((y_train.shape[0], y_train.shape[1]))
max_iteration = 7
theta = S+B
for j in range(max_iteration):
    for i in range(len(X_train)):
        #print(X_train[i].shape)
        theta = S+B
        hypothesis = (np.dot((np.transpose(theta)), X_train[i]))
        #print(hypothesis.reshape(8,1))
        activation[i] = (hypothesis)
        #print(activation.shape)
#        loss =  logistic_regression_loss(y_train[i], activation[i], S, B, 0.01, 0.02, len(X_train))
#        print("This is losss",loss)
    log_loss_1 = log_loss(y_train, activation)
    #print(log_loss_1)
    total_loss = log_loss_1 + norm_11(S)+ norm_1_infinity(B)
    print(total_loss)
    gradient_1 = gradiant_calculation(activation, X_train, y_train, len(X_train))
    gradient_1 -= (alpha*gradient_1)/(len(X_train))
    theta = gradient_1
    update_S_norm = proximal_norm_11(S, lambda_1, 0.00000000001, gradient_1)
    update_B_norm = proximal_norm_1_infinity(B, lambda_2, 0.00000000001, gradient_1)
    S = update_S_norm
    B = update_S_norm
    
    
    #gradient_S = np.zeros((S.shape[0], S.shape[1]))
    #gradient_B = np.zeros((B.shape[0], B.shape[1]))
    #for j in range(len(B)):
     #   gradient_S = (1/len(X_train))*()
        
    