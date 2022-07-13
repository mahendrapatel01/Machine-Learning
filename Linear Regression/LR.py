#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:01:56 2020

@author: 203050078
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

np.random.seed(42)
#references : https://www.geeksforgeeks.org/standardscaler-minmaxscaler-and-robustscaler-techniques-ml/
class Scaler():
    # hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    def __init__(self):
        pass
    def __call__(self,features, is_train=False):
        #normalization scaling
        #e_min = np.min(features,axis=0)
        #e_max = np.max(features,axis=0)
        #features = features - e_min
        #features = features / (e_max-e_min)
        
        #standardization scaling
        #e_mean = np.mean(features,axis=0)
        #e_std = np.std(features,axis=0)
        #features = (features - e_mean)/e_std
        for i in range(len(features[0])):
            #min-max scaler
           features[:,i]=features[:,i]-np.min(features[:,i])
           features[:,i]=features[:,i]/(np.max(features[:,i])-np.min(features[:,i])) 
           features[:,i]=(features[:,i]-np.mean(features[:,i],axis=0))/np.std(features[:,i],axis=0)
        #   pass
       
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
def get_features(csv_path,is_train=False,scaler=None):
    '''
    Description:
    read input feature columns from csv file
    manipulate feature columns, create basis functions, do feature scaling etc.
    return a feature matrix (numpy array) of shape m x n 
    m is number of examples, n is number of features
    return value: numpy array
    '''

    '''
    Arguments:
    csv_path: path to csv file
    is_train: True if using training data (optional)
    scaler: a class object for doing feature scaling (optional)
    '''
    dataset = pd.read_csv(csv_path)
    dataset.drop(columns=[' shares'],inplace=True)
    dataset = dataset.to_numpy()
    if(is_train and scaler!=None):
        scaler(dataset)
        
    ones = np.ones((len(dataset),1))   
    dataset = np.append(ones,dataset,axis=1)
    return dataset

#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
def get_targets(csv_path):
    '''
    Description:
    read target outputs from the csv file
    return a numpy array of shape m x 1
    m is number of examples
    '''      
    dataset = pd.read_csv(csv_path,usecols=[' shares'])
    return dataset.to_numpy()


def analytical_solution(feature_matrix, targets, C=0.0):
    '''
    Description:
    implement analytical solution to obtain weights
    as described in lecture 5d
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    '''
    x = feature_matrix
    y = targets
    xt = np.transpose(x)
    n = len(x[0])
    m = len(x)
    iden = np.identity(n,dtype='float')
    ws = np.matmul(np.matmul(np.matrix(np.matmul(xt,x) + (m*C*iden)).I,xt),y)  
    return ws

def get_predictions(feature_matrix, weights):
    '''
    description
    return predictions given feature matrix and weights
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    '''
    return np.matmul(feature_matrix,weights)

def mse_loss(feature_matrix, weights, targets):
    '''
    Description:
    Implement mean squared error loss function
    return value: float (scalar)
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    '''
    return np.sum(np.square(np.matmul(feature_matrix, weights)-targets))/len(targets)
    
def l2_regularizer(weights):
    '''
    Description:
    Implement l2 regularizer
    return value: float (scalar)
    '''

    '''
    Arguments
    weights: numpy array of shape n x 1
    '''
    return np.sum(np.square(weights))

def loss_fn(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute the loss function: mse_loss + C * l2_regularizer
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: float (scalar)
    '''
    return mse_loss(feature_matrix, weights, targets) + C * l2_regularizer(weights)

#https://medium.com/analytics-vidhya/linear-regression-in-python-from-scratch-24db98184276
def compute_gradients(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute gradient of weights w.r.t. the loss_fn function implemented above
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: numpy array
    '''
    return 2*(np.matmul(np.transpose(feature_matrix),(np.matmul(feature_matrix,weights)-targets))/len(targets) + C*weights)


#https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html    
def sample_random_batch(feature_matrix, targets, batch_size):
    '''
    Description
    Batching -- Randomly sample batch_size number of elements from feature_matrix and targets
    return a tuple: (sampled_feature_matrix, sampled_targets)
    sampled_feature_matrix: numpy array of shape batch_size x n
    sampled_targets: numpy array of shape batch_size x 1
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    batch_size: int
    '''
    narr = np.append(feature_matrix,targets,axis=1)
    np.random.shuffle(narr)
    return (narr[0:batch_size,0:-1],narr[0:batch_size,-1:])

def initialize_weights(n):
    '''
    Description:
    initialize weights to some initial values
    return value: numpy array of shape n x 1
    '''

    '''
    Arguments
    n: int
    '''
    return np.zeros((n,1))

def update_weights(weights, gradients, lr):
    '''
    Description:
    update weights using gradient descent
    retuen value: numpy matrix of shape nx1
    '''

    '''
    Arguments:
    # weights: numpy matrix of shape nx1
    # gradients: numpy matrix of shape nx1
    # lr: learning rate
    ''' 
    return weights - (lr * gradients)


def early_stopping(arg_1=None, arg_2=None, arg_3=None, arg_n=None,arg_5=5,arg_6=5):
    # allowed to modify argument list as per your need
    # return True or False
    #raise NotImplementedError
    max_cnt_err = 10
    arg_1=int(arg_1)
    arg_2=int(arg_2)
    arg_3=int(arg_3)
    arg_n=int(arg_n)
    if(arg_3 > arg_n):
        if(arg_5 == max_cnt_err):
            #print("t1")
            return True
    
    if(arg_1 > arg_2):
        if(arg_6 == max_cnt_err):
            #print("t2")
            return True
    
    return False
    
    
def do_gradient_descent(train_feature_matrix,  
                        train_targets, 
                        dev_feature_matrix,
                        dev_targets,
                        lr=1.0,
                        C=0.0,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=100):
    '''
    feel free to significantly modify the body of this function as per your needs.
    ** However **, you ought to make use of compute_gradients and update_weights function defined above
    return your best possible estimate of LR weights
    a sample code is as follows -- 
    '''
    n = len(train_feature_matrix[0])
    weights = initialize_weights(n)
    wgt = initialize_weights(n)
    #weights = analytical_solution(train_feature_matrix,train_targets)
    dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)
    #variable for early stopping
    der1,ter1 = dev_loss,train_loss
    cnt_terr=0
    cnt_derr=0
    min_wt = wgt
    min_err = train_loss
    ###
    print("step {} \t dev loss: {} \t train loss: {}".format(0,dev_loss,train_loss))
    for step in range(1,max_steps+1):

        #sample a batch of features and gradients
        features,targets = sample_random_batch(train_feature_matrix,train_targets,batch_size)
        
        #compute gradients
        gradients = compute_gradients(features, weights, targets, C)
        
        #update weights
        wgt = weights 
        weights = update_weights(weights, gradients, lr)

        if step%eval_steps == 0:
            dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
            train_loss = mse_loss(train_feature_matrix, weights, train_targets)
            print("step {} \t dev loss: {} \t train loss: {}".format(step,dev_loss,train_loss))
            if(min_err > train_loss):
                min_err = train_loss
                min_wt = np.array(weights)
            if( train_loss > ter1 ):
                cnt_terr +=1
                print("greater t")
            else:
                cnt_terr=0
                
            if( dev_loss > der1 ):
                cnt_derr +=1
                print("greater d")
            else:
                cnt_derr=0
                
            if( early_stopping( dev_loss, der1, train_loss, ter1, cnt_terr,cnt_derr)):
                return min_wt
                print("early stooping")
            der1 = dev_loss
            ter1 = train_loss
        
        '''
        implement early stopping etc. to improve performance.
        '''
    print("gradient weights : ",weights)
    print("min_error : ",min_err)
    return weights    

def do_evaluation(feature_matrix, targets, weights):
    # your predictions will be evaluated based on mean squared error 
    predictions = get_predictions(feature_matrix, weights)
    loss =  mse_loss(feature_matrix, weights, targets)
    return loss


if __name__ == '__main__':
    scaler = Scaler() #use of scaler is optional
    train_features, train_targets = get_features('data/train.csv',True,scaler), get_targets('data/train.csv')
    dev_features, dev_targets = get_features('data/dev.csv',True,scaler), get_targets('data/dev.csv')

    a_solution = analytical_solution(train_features, train_targets, C=1e-8)
    print('evaluating analytical_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, a_solution)
    train_loss=do_evaluation(train_features, train_targets, a_solution)
    print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
    print('training LR using gradient descent...')
    gradient_descent_soln = do_gradient_descent(train_features, 
                        train_targets, 
                        dev_features,
                        dev_targets,
                        lr=0.00001,
                        C=1e-8,
                        batch_size=100,
                        max_steps=2000000,
                        eval_steps=50)

    print('evaluating iterative_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss=do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
    
    #generating test predictions
    
    #test_features = pd.read_csv('data/test.csv',)
    #test_features = test_features.to_numpy()
    #ones = np.ones((len(test_features),1))   
    #test_features = np.append(ones,test_features,axis=1)
   # test_features = np.append(get_features('data/test.csv',False,scaler),get_targets('data/test.csv'),axis=1)
    #out_shares = get_predictions(test_features, a_solution)
    
    #print(out_shares)
    #np.savetxt("output.csv",[i for i in enumerate(out_shares)],delimiter=',',fmt = ['%d','%e'],header='instance_id,shares',comments='')