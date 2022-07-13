#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 01:01:10 2020

@author: mahi
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

num_input = 90
num_output=1

class Net(object):

    def relu(self,Z):
        return np.maximum(0,Z)

    def __init__(self, num_layers, num_units):

        num_layers=num_layers+1
        self.num_layers = num_layers
        self.num_units = num_units
        np.random.seed(42)
        self.weights=[]
        self.bias=[]
        for i in range(num_layers):
            if(i==0):
                self.weights.append(np.random.uniform(-1,1,size=(num_input,num_units)))
                self.bias.append(np.random.uniform(-1,1,size=(1,num_units)))
            elif( i==num_layers-1):
                self.weights.append(np.random.uniform(-1,1,size=(num_units,num_output)))
                self.bias.append(np.random.uniform(-1,1,size=(1,num_output)))
            else:
                self.weights.append(np.random.uniform(-1,1,size=( num_units,num_units)))
                self.bias.append(np.random.uniform(-1,1,size=(1,num_units)))

    def __call__(self, X):

        inp = X
#         print(inp.shape)
        self.net_out=[]
        self.g_out=[]
        for i in range(self.num_layers):
            z = inp.dot(self.weights[i])+self.bias[i]
            self.net_out.append(z)
#             print('z : ',z.shape)
            if(i<self.num_layers-1):
                inp = self.relu(z)
                self.g_out.append(inp)
            else:
                self.g_out.append(z)


    def backward(self, X, y, lamda):

        def der_relu(x):
            y=np.array(x)
            y[y<=0]=0
            y[y>0]=1
            return y

        delta_wt=[x for x in range(self.num_layers)]
        delta_bias=[x for x in range(self.num_layers)]
        delta_A=[x for x in range(self.num_layers)]
        delta_Z=[x for x in range(self.num_layers)]

        for i in range(self.num_layers-1,-1,-1):
            if(i==self.num_layers-1):
                delta_A[i]= 2*np.subtract(y,self.g_out[i])/len(y)
                delta_Z[i]= delta_A[i]
            else:
                delta_A[i]=delta_Z[i+1].dot(self.weights[i+1].T)/len(delta_Z[i+1])
                delta_wt[i+1]=self.g_out[i].T.dot(delta_Z[i+1]) + lamda*self.weights[i+1]
                delta_bias[i+1]=np.sum(delta_Z[i+1],axis=0)
                delta_Z[i]=delta_A[i]*der_relu(self.net_out[i])

            delta_wt[0]=X.T.dot(delta_Z[1])
            delta_bias[0]=np.sum(delta_Z[0],axis=0)

        return (delta_wt,delta_bias,delta_A,delta_Z)



class Optimizer(object):


    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


    def step(self, weights, biases, delta_weights, delta_biases):

        for i in range(len(weights)):
#             print(delta_weights[i])
            weights[i] = np.add(weights[i] , (self.learning_rate*delta_weights[i]))

        for i in range(len(weights)):
#             print(delta_biases[i])
            biases[i] = np.add(biases[i] , (self.learning_rate*delta_biases[i]))

        return (weights,biases)

def loss_mse(y, y_hat):
#     print('y',np.shape(y))
#     print('y_hat',np.shape(y_hat))

    return np.sum(np.square(y-y_hat))/len(y)

def loss_regularization(weights, biases):
    sum=0.0
    for i in range(len(weights)):
        sum+=np.sum(np.square(weights[i]))+np.sum(np.square(biases[i]))
    return sum

def loss_fn(y, y_hat, weights, biases, lamda):
    return loss_mse(y,y_hat)+lamda*loss_regularization(weights,biases)

def rmse(y, y_hat):
    return np.sqrt(loss_mse(y,y_hat))

def get_batch(feature_matrix,targets,batch_size,batch_no):
#     narr = np.append(feature_matrix,targets,axis=1)
#np.random.shuffle(narr)
    l = batch_no*batch_size
#     return (narr[l:l+batch_size,0:-1],narr[l:l+batch_size,-1:])
    return feature_matrix[l:l+batch_size],targets[l:l+batch_size]

def get_test_data_predictions(net, inputs):
    '''
    Perform forward pass on test data and get the final predictions that can
    be submitted on Kaggle.
    Write the final predictions to the part2.csv file.
    Parameters
    ----------
        net : trained neural network
        inputs : test input, numpy array of shape m x d
    Returns
    ----------
        predictions (optional): Predictions obtained from forward pass
                                on test data, numpy array of shape m x 1
    '''
    net(inputs)
   # print(net.g_out[-1].shape)
    return np.round(net.g_out[-1]).reshape((len(net.g_out[-1]),1))

def train(
    net, optimizer, lamda, batch_size, max_epochs,
    train_input, train_target,
    dev_input, dev_target
):
    terror=[]
    derror=[]
    for i in range(max_epochs):
        l = len(train_input)//batch_size
        for j in range(l):
            features,targets = get_batch(train_input,train_target,batch_size,j)
            net(features)
            delta_wt,delta_bias,delta_A,delta_Z=net.backward(features,targets,lamda)
            optimizer.step(net.weights,net.bias,delta_wt,delta_bias)

        terror.append(rmse(targets, net.g_out[-1]))
        net(dev_input)
        derror.append(rmse(dev_target, net.g_out[-1]))
#         print("gout shape : ",net.g_out[-1].shape)
        #print(i,'- train_error : ',terror[i],'  ,  dev_error : ',derror[i])
   # print(i,'- train_error : ',terror[max_epochs-1],'  ,  dev_error : ',derror[max_epochs-1])

    return terror,derror

def read_data():
    dataset = pd.read_csv('/home/mahi/Desktop/iit_bombay/FML/assignment2/dataset/train.csv')
    inp = np.array(dataset.drop(columns=['label']))
    opt = np.array(dataset['label'])
    dev_dataset = pd.read_csv('/home/mahi/Desktop/iit_bombay/FML/assignment2/dataset/dev.csv')
    inp_dev = np.array(dataset.drop(columns=['label']))
    opt_dev = np.array(dataset['label'])
    opt.reshape((len(opt),1))
    opt_dev.reshape((len(opt_dev),1))

    inp_test = pd.read_csv('/home/mahi/Desktop/iit_bombay/FML/assignment2/dataset/test.csv')
    inp_test=np.array(inp_test)
    return inp, opt, inp_dev, opt_dev, inp_test



def main():
    max_epochs = 50
    batch_size = 128


    learning_rate = 0.001
    num_layers = 1
    num_units = 64
    lamda = 5 # Regularization Parameter

    train_input, train_target, dev_input, dev_target, test_input = read_data()
    net = Net(num_layers, num_units)
    optimizer = Optimizer(learning_rate)
    train(
        net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target.reshape((len(train_target),1)),
        dev_input, dev_target.reshape((len(dev_target),1))
    )
    ans=get_test_data_predictions(net, test_input)
    Id=[]
    for i in range(1,len(ans)+1):
        Id.append(i)

  #  print("ans : ",ans)
    res =pd.DataFrame(np.column_stack([Id,ans]),columns=['Id','Predicted'])
    res.to_csv("predict_1.csv",index=False)


if __name__ == '__main__':
    main()

