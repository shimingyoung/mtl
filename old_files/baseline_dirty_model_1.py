import numpy as np
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import MultiTaskLassoCV
import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from scipy import linalg
from sklearn.metrics import log_loss
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from scipy import linalg
input_data = pd.read_csv("onpoint5_outcomes_features_02102018.csv")


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


def sigmoid_activation(hypothesis):
    hypothesis = np.float_(hypothesis)
    a = np.zeros((len(hypothesis),))
    for i in range(len(hypothesis)):
        a[i] = (1/(1+np.exp(-hypothesis[i])))
        if a[i] >= 0.5:
            a[i] = 1
        else:
            a[i] = 0
    return(a)

def logistic_regression(y_label, activation, N):
	loss = 1/N*(np.sum(np.multiply(-1*(y_label), np.log(activation))) - np.multiply((1 - y_label), np.log(1 - activation)))
	return(loss)


def gradient_logistic_function(activation, X_train, y_train):
	return(np.dot(np.transpose(X_train), activation - y_train))

def precision(y_true, activation):
    precision_scores = []
    for i in range(y_true.shape[1]):
        #print(y_true[:, i])
        p1 = precision_score(y_true[:, i], activation[:, i])
        precision_scores.append(p1)
    return(precision_scores)
    
def recall(y_true, activation):
    recall_scores = []
    for i in range(y_true.shape[1]):
        a = y_true[:, i]
        b = activation[:, i]
        r1 = recall_score(a,b, average='micro')
        recall_scores.append(r1)
    return(recall_scores)
    
def roc_auc_score_1(y_true, activation):
    roc_auc_scores = []
    for i in range(y_true.shape[1]):
        a = y_true[:, i]
        b = activation[:, i]
        r1 = roc_auc_score(a,b, average='macro')
        roc_auc_scores.append(r1)
    return(roc_auc_scores)


def roc_1(y_true, activation):
    roc_scores = []
    for i in range(y_true.shape[1]):
        a = y_true[:, i]
        b = activation[:, i]
        r1 = roc_curve(a,b)
        roc_scores.append(r1)
    return(roc_scores) 

def predict(S,X_test):
    activation = np.zeros((X_test.shape[0], 8))
    theta = S
    for i in range(len(X_test)):
        hypothesis = (np.dot((np.transpose(theta)), X_train[i]))
        activation[i] = sigmoid_activation(hypothesis)
    return(activation)


def project_L1_ball(D, tau): #TODO verify
    # min (1/2) ||X-C||_2^2, s.t. ||X||_1 <= t
    q = np.sign(D)
#    print(q)
    a = np.zeros((D.shape[0], D.shape[1]))
    for i in range(len(D)):
        for j in range(D.shape[1]):
            a[i][j] = np.maximum(0, (abs(D[i][j])-tau))
    return np.multiply(a,q)





def condition_check(f_x, f_x_1):
	if f_x <= f_x_1:
		return(True)
	else:
		return(False)


def second_order_expansion(loss_old , weight,  weight_old, gradient_old, L, lambda_1):
	a = loss_old
	b1 = weight - weight_old
	b1 = b1.flatten()
	b2 = gradient_old
	b2 = b2.flatten()
	b = np.inner(b1, b2)
	#print(b.shape)
	c = lambda_1*norm_11(weight_old)
	d = L/2*np.linalg.norm((weight - weight_old), ord=2)
	return(a+b+c+d)

def first_condition(weight_old, X_train, y_train, lambda_1):
	activation = np.zeros((y_train.shape[0], y_train.shape[1]))
	for i in range(len(X_train)):
		hypothesis = np.dot(np.transpose(weight_old), X_train[i])
		activation[i] = sigmoid_activation(hypothesis)
	c = log_loss(y_train, activation)
	loss =  c + lambda_1*norm_11(weight_old)
	gradient_updated = gradient_logistic_function(activation, X_train, y_train)
	#print(loss)
	return(loss, c, gradient_updated)


test_data, feature_data, test_label_1 = test_label(input_data)
X = np.array(feature_data)
X = linalg.block_diag(X)
for i in range(len(X)):
	for j in range(X.shape[1]):
		X[i][j] = 0
y = np.array(test_data)
X = normalize(X, norm='l2', axis =1, copy=True, return_norm=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 43)
print(X_train.shape)
N = len(X_train)
lambda_1 = 0.9
lambda_2 = 0.01
L_0 = 25
t = 1/L_0
eta = 100
eta_old = 0.02
loss_old = 0
S = np.random.normal(0,1,(X_train.shape[1], y_train.shape[1]))
print(S.shape)
#B = np.random.normal(0,1,(X_train.shape[1], y_train.shape[1]))
activation = np.zeros((y_train.shape[0], y_train.shape[1]))
max_iteration = 20
gradient_old = np.zeros((X_train.shape[1], y_train.shape[1]))
for j in range(max_iteration):
	for p in range(len(X_train)):
		hypothesis = np.dot((np.transpose(S)), X_train[p])
		activation[p] = sigmoid_activation(hypothesis)
	log_loss_1 = log_loss(y_train, activation)
	loss = log_loss_1 + lambda_1*norm_11(S)
	print(loss)
	gradient_new = gradient_logistic_function(activation, X_train, y_train)
	if loss_old == loss:
		break
	for i in range(16):
		if i == 0:
			L_new = eta**i*L_0
		else:
			L_new = eta**i*L_old
		S_bt = project_L1_ball(S - t*gradient_new, lambda_1*t)
		S_updated = S_bt - S
		a, c, gradient_old = first_condition(S_updated, X_train, y_train, lambda_1)
		#print(a)
		if i == 0:
			gradient_1 = gradient_new
		else:
			gradient_1 = gradient_old
		b = second_order_expansion(c, S_updated, S, gradient_1, L_new, lambda_1)
		#print(b)
		if a >= b:
			break
		else:
			S = S_updated
		L_old = L_new
		gradient_new = gradient_1
	t = 1/L_new
	t_new = (1 + np.sqrt(1 + 4 *(t**2)))/2
	eta_1= (t-1)/t_new
	if j == 0:
		S_old = 0
	else:
		S = S + eta_1*(S - S_old)
	S_old = S
	loss_old = loss	
output_activation = predict(S, X_test)
precision_score_list = precision(y_test, output_activation)
print(precision_score_list)
recall_score_list = recall(y_test, output_activation)
print(recall_score_list)
roc_auc_score_list = roc_auc_score_1(y_test, output_activation)
print(roc_auc_score_list)
roc_curve_list = roc_1(y_test, output_activation)



	








# for j in range(max_iteration):
# 	for i in range(len(X_train)):
# 		# print(X_train[i].shape)
# 		# print(S.shape)
# 		hypothesis = np.dot((np.transpose(S)), X_train[i])
# 		activation[i] = sigmoid_activation(hypothesis)
# 	log_loss_1 = log_loss(y_train, activation)
# 	loss_old = loss
# 	loss = log_loss_1 + lambda_1*norm_11(S)
# 	if loss_old == loss:
# 		break
# 	print(loss)
# 	gradient_1 = gradient_logistic_function(activation)
# 	# print(gradient_1.shape)
# 	S_old = S - eta*gradient_1
# 	eta = (eta/eta_old)*i
# 	# print(S_old.shape)
# 	# print(S.shape)
# 	S = project_L1_ball(S_old, lambda_1/L)
# 	# print(S.shape)
# output_activation = predict(S, X_test)
# precision_score_list = precision(y_test, output_activation)
# print(precision_score_list)
# recall_score_list = recall(y_test, output_activation)
# print(recall_score_list)
# roc_auc_score_list = roc_auc_score_1(y_test, output_activation)
# print(roc_auc_score_list)
# roc_curve_list = roc_1(y_test, output_activation)

