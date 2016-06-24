# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle
import matplotlib.pyplot as plt
import random

def initializeWeights(n_in, n_out):

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon;
    return W


def sigmoid(z):
        
    s = 1 / (1 + np.exp(-z))
    return s 


def preprocess():

    #loading data set available in mnist_all.mat file
    mat = loadmat("C:\Users\Python\mnist_all.mat")  # loads the MAT object as a Dictionary

    A = mat.get('train0')
    B = mat.get('test0')
    a = range(A.shape[0])
    aperm = np.random.permutation(a)
    validation_data = np.double(A[aperm[0:1000],:])/255
    train_data = np.double(A[aperm[1000:],:])/255
    test_data = np.double(B)/255
    
    train_label = np.zeros((train_data.shape[0],10))
    train_label[:,0] = 1;
    
    validation_label = np.zeros((validation_data.shape[0],10))
    validation_label[:,0] = 1;
    
    test_label = np.zeros((test_data.shape[0],10))
    test_label[:,0] = 1;
    
    for i in range(1,10):
        A = mat.get('train'+str(i))
        B = np.double(mat.get('test'+str(i)))/255
        a = range(A.shape[0])
        aperm = np.random.permutation(a)
        A1 = np.double(A[aperm[0:1000],:])/255
        A2 = np.double(A[aperm[1000:],:])/255
        
        validation_data = np.concatenate((validation_data,A1),axis=0) 
        train_data = np.concatenate((train_data, A2),axis=0)
        
        temp_training_label = np.zeros((A2.shape[0],10))
        temp_training_label[:,i] = 1;
        train_label = np.concatenate((train_label , temp_training_label))

        temp_validation_label = np.zeros((A1.shape[0],10))
        temp_validation_label[:,i] = 1;
        validation_label = np.concatenate((validation_label , temp_validation_label))
        
        test_data = np.concatenate((test_data, B),axis=0)
        
        temp_test_label = np.zeros((B.shape[0],10))
        temp_test_label[:,i] = 1;
        test_label = np.concatenate((test_label , temp_test_label))
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    #50X785
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    #10X51
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    
    # ***** Step 1 - calculation for Zj (Eqn 2) *****
    temp_training_data = np.concatenate((training_data ,np.ones((training_data.shape[0], 1)) ),axis=1) 
    temp_training_data = np.transpose(temp_training_data)
    #50*50000
    Z = sigmoid(np.dot(w1 , temp_training_data))
    
    
    # ***** Step 2 - Calculation for Ol (Eqn 4) *****
    # 51X50000
    Z_Bias = np.concatenate((Z ,np.ones((1, Z.shape[1])) ),axis=0) 
    O = sigmoid(np.dot(w2 , Z_Bias))
    
    
    # ***** Step 3 - Calculation for Delta ( Eqn 9) *****
    training_label_transpose = np.transpose(training_label)
    # (Eqn 9)10X50000
    deltaL = O - training_label_transpose
        
    # ***** Step 4 - Calculation for obj_val using Eqn 5 and 15 *****
    print "-----------"
    # Calculation for regularization value
    reg = (lambdaval * (np.sum(w1 * w1) + np.sum(w2 * w2)))/(2 * training_label.shape[0])
    # Final obj_val
    A = np.sum((Clabels - Tlabels)**2 for (Clabels,Tlabels) in zip( O, np.transpose(training_label)))
    obj_val = np.sum(A)/training_label.shape[0]
    obj_val = obj_val + reg
    print obj_val
    
    
    # ***** Step 5 - Calculation for grad_w2 ( Eqn 9 and 16 ) *****
    grad_w2 = (np.dot(deltaL , np.transpose(Z_Bias)) + lambdaval * w2)/training_label.shape[0]
    
    
    # ***** Step 6 - Calculation for grad_w1 ( Eqn 12 and 17 ) *****
    Z_Bias = np.transpose(Z)
    w2 = w2[:,range(0,w2.shape[1]-1)]
    iCal_1 = ((1 - Z) * Z) * ( np.dot( np.transpose(w2), deltaL ))
    iCal_2 = np.dot(iCal_1 , np.transpose(temp_training_data))
    iCal_2 = np.transpose( iCal_2 )
    #w1 = w1[:,range(0,w1.shape[1]-1)]
    iCal_3 = lambdaval * np.transpose(w1)
    grad_w1 = (iCal_2 + iCal_3)/training_data.shape[0]
    grad_w1 = np.transpose(grad_w1)
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    return (obj_val,obj_grad)

def nnPredict(w1, w2, data):

    data = np.concatenate((data ,np.ones((data.shape[0], 1)) ),axis=1) 
    data = np.transpose(data)
    Z = sigmoid(np.dot(w1 , data))
    Z_Bias = np.concatenate((Z ,np.ones((1, Z.shape[1])) ),axis=0) 
    O = sigmoid(np.dot(w2 , Z_Bias))
    
    O = np.transpose(O)
    
    # Calculating label
    temp = O.argmax(axis=1)
    temp = np.transpose(temp)
    return temp

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess();


# Training Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1];

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;

# set the number of nodes in output unit
n_class = 10;

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0.5;

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

predicted_label = nnPredict(w1,w2,train_data)

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label.argmax(axis=1)).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label.argmax(axis=1)).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label.argmax(axis=1)).astype(float))) + '%')

pickle.dump([n_hidden,w1,w2],open("params.pickle","wb"))

p = pickle.load(open("C:\Users\Python\params.pickle","rb"))