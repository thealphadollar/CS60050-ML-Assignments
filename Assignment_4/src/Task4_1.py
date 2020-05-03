#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:37:39 2020

@author: onebyteatatime
"""
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import pandas as pd
from random import *
import matplotlib.pyplot as plt
datafile= open('../data/seeds_dataset.txt')


input_layer = 8
output_layer = 3


def sigmoid(x):
    return (1/(1 + np.exp(-x)))


def gradient_sigmoid(x):
    return sigmoid(x)*sigmoid(1-x);


def preprocessing(datafile):
    dataset = []
    for l in datafile:
        thisL = list(map(float,l.split()));
        dataset.append(thisL)
        
    dataset = np.asarray(dataset)
    attribute= stats.zscore(dataset[:, :7], axis = 0)
    label = dataset[:,7].astype(int)
    one_hot = np.zeros((label.size, label.max()))
    one_hot[np.arange(label.size),label-1] =1;
    final_dataset = np.concatenate((attribute, one_hot), axis = 1)
    train, test = train_test_split(final_dataset, test_size = 0.2, random_state = 24)
    pd.DataFrame(train).to_csv("../data/train_dataset.csv", header = None, index= None)
    pd.DataFrame(test).to_csv("../data/test_dataset.csv", header = None, index = None)
    print(train.shape)
    print(test.shape)
    return train, test

    
def data_loader(train_dataset):
    i = 0
    batches = []
    print(train_dataset.shape[0])
    while i < train_dataset.shape[0] :
        batches.append(train_dataset[i: i+32])
        i += 32
    return batches


def weight_initialiser(n, m):
    return (np.random.rand(n, m) - 0.5)/0.5

def forward_prop(weights, left_out, bias):
    #print("Weights Shape ", weights.shape)
    #print("X shape", left_out.shape)
    z = (left_out).dot(weights) + bias
    return z


def backward_prop(deltaRight, weights, activationRight):
    delta = deltaRight.dot(weights.T)*(1 - np.power(activationRight, 2))
    return delta

def soft_max(z2):
    exp_score = np.exp(z2);
    return exp_score/np.sum(exp_score, axis = 1, keepdims = True)

def cross_entropy(pred, real):
    # print(pred.shape, real.shape)
    num = real.shape[0]
    return (pred-real)/num


def printCost(train, w1, w2, b1, b2):
    num   = train.shape[0]
    feature = train[:, 0:7]
    label = train[:, 7:]
    z1 = forward_prop(w1, feature, b1)
    a1 = sigmoid(z1)
    z2 = forward_prop(w2, a1, b2)
    a2 = soft_max(z2)
    for i in a2 :
        maxVal= np.max(i)
        if(i[0] == maxVal): 
            i[0] = 1
            i[1] = 0
            i[2] = 0
        if(i[1] == maxVal):
            i[1] = 1
            i[0] = 0
            i[2] = 0
        if(i[2] == maxVal):
            i[2] = 1
            i[0] = 0
            i[1] = 0
    count = 0
    for i in range(a2.shape[0]):
        if(a2[i][0] == label[i][0] and a2[i][1] == label[i][1] and a2[i][2] == label[i][2]):count+=1
        
    return count*100/a2.shape[0];
    

def training(train, test, batches, num_epochs, learning_rate):
    w1 = weight_initialiser(7, 32)
    b1 = np.zeros((1, 32))
    w2 = weight_initialiser(32, 3)
    b2 = np.zeros((1, 3))
    test_acc = []
    train_acc= []
    finMod = {}
    itr =[]
    for i in range(num_epochs):
        for batch in batches:
            feature= batch[:,0:7]
            label = batch[:,7: ]
            #Forward Prop
            z1 = forward_prop(w1, feature, b1)
            a1 = sigmoid(z1)
            #print("A1 Shape ", a1.shape)
            z2 = forward_prop(w2, a1, b2)
            a2 = soft_max(z2)
            #Backward Prop
            delta3 = cross_entropy(a2, label)
            dw2 = (a1.T).dot(delta3)
            db2 =np.sum(delta3, axis = 0, keepdims = True)
            delta2 = delta3.dot(w2.T)*(1- np.power(a1, 2))
            dw1= (feature.T).dot(delta2)
            db1 = np.sum(delta2, axis = 0)
            
            w1 -= learning_rate * dw1
            w2 -= learning_rate * dw2
            b1 -= learning_rate * db1
            b2 -= learning_rate * db2
                  
        if(i%10 == 0):
            train_acc.append(printCost(train,w1, w2, b1, b2))
            test_acc.append(printCost(test, w1, w2, b1, b2))
            itr.append(i);
            print('Cost for Train', printCost(train, w1, w2, b1, b2))
            print('Cost of Test', printCost(test, w1, w2, b1, b2))
    finMod = {'W1': w1, 'B1': b1, 'W2': w2, 'B2': b2}
    return finMod, train_acc, test_acc, itr

train, test = preprocessing(datafile)
#print(train.shape)
batches = data_loader(train)
#training(batches)
model, train_acc, test_acc, itr = training(train, test, batches, 200, 0.01)

plt.plot( itr, train_acc)
plt.xlabel('Cost')
plt.ylabel('Iterations')
plt.savefig('../TrainAccVSIterations.png')
plt.clf()

plt.plot( itr, test_acc)
plt.xlabel('Cost')
plt.ylabel('Iterations')
plt.savefig('../TestAccVSIterations.png')





