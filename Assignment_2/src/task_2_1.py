import numpy as np
import pandas as pd
from os import path
import json

def g(inpu):
    return (1/(1+np.exp(-inpu)))

def cost(inpu, actual):
    return (actual*np.log(inpu)) + ((1-actual)*(np.log(1-inpu)))

class LogisticRegressor():
    def __init__(self, lr=0.05):
        # store learning rate and weights
        self.lr = lr
        self.weights = None

    def fit(self, train_data_feature, train_data_label):
        # get shape of the training data
        m,n = train_data_feature.shape

        # creating numpy vector with feature data
        X = np.zeros(shape=(m,n+1))
        for i in range(m):
            for j in range(n+1):
                if j==0:
                    X[i][j] = 1
                else:
                    X[i][j] = train_data_feature[i][j-1]
        # setting initial value of the parameters as 1
        coeff_vals = np.ones(n+1)
        # stores previous cost
        prev_jtheta = None
        # calculate loss
        h_theta = g(np.dot(X, coeff_vals))
        loss = (h_theta - train_data_label)
        # current cost
        cur_jtheta = np.sum(cost(h_theta, train_data_label)) * (-1/m)
        # setting convergence when the difference between consecutive error is less than 0.0000001
        while (prev_jtheta is None or abs(prev_jtheta - cur_jtheta) > 0.0000001):
            # gradient descent with vector notation, simultaneous calculation
            descent_vals = (np.dot(X.transpose(), loss) * self.lr) / m
            # update all coefficients with descent
            coeff_vals -= descent_vals
            prev_jtheta = cur_jtheta
            # calculate new cost
            h_theta = g(np.dot(X, coeff_vals))
            loss = (h_theta - train_data_label)
            cur_jtheta = np.sum(cost(h_theta, train_data_label)) * (-1/m)
            print(f"Difference between consecutive costs: {abs(prev_jtheta - cur_jtheta)}\t", end="\r", flush=True)
        # print(f"Parameters: {list(coeff_vals)}\t\t")
        # print(f"Error on Data: {cur_jtheta}\n")
        
        self.weights = coeff_vals

    # function for predicting results
    def predict(self, data_feature):
        m,n = data_feature.shape
        # creating numpy vector with feature data
        X = np.zeros(shape=(m,n+1))
        for i in range(m):
            for j in range(n+1):
                if j==0:
                    X[i][j] = 1
                else:
                    X[i][j] = data_feature[i][j-1]
        h_theta = g(np.dot(X, self.weights))
        # threshold is chosen as 0.5
        h_theta[h_theta>=0.5] = 1
        h_theta[h_theta<0.5] = 0
        return h_theta

    def get_params(self, deep = False):
        return {'lr':self.lr}
