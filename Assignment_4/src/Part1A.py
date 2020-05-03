#!/usr/bin/env python
# coding: utf-8
# @author: Shivam Kumar Jha
# @rollno: 17CS30033

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork:
    
    def __init__(self,
                num_hidden,
                num_neurons_per_layer,
                activation_func_hidden,
                num_neurons_out_layer=3,
                activation_func_output="softmax",
                opt_algo="SGD",
                loss_func="categorial_cross_entropy_loss",
                learning_rate=0.01,
                num_epochs=200):
        self.num_hidden = num_hidden
        self.num_neurons_per_layer = num_neurons_per_layer
        self.activation_func_hidden = activation_func_hidden
        self.num_neurons_out_layer = num_neurons_out_layer
        self.activation_func_output = activation_func_output
        self.opt_algo = opt_algo
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
    
    def fit(self):
        self.dataLoader()
        self.train()
    
    def plot(self):
        iterations = [x for x in range(1,201,10)]
        plt.plot(iterations, self.train_accuracy, label="Train")
        plt.plot(iterations, self.test_accuracy, label="Test")
        plt.xlabel('Cost')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title("Part 1A Accuracy Vs Iterations")
        plt.show()
        # plt.savefig('../plots/part1a.png')
        plt.clf()
        
    def train(self):
        self.test_accuracy = list()
        self.train_accuracy = list()

        weight_l1 = self.weightInitialiser(7, self.num_neurons_per_layer) 
        weight_l2 = self.weightInitialiser(self.num_neurons_per_layer, self.num_neurons_out_layer)

        blayer1 = np.zeros((self.num_hidden, self.num_neurons_per_layer))
        blayer2 = np.zeros((self.num_hidden, self.num_neurons_out_layer))

        for cur_epoch in range(self.num_epochs):
            for feature, label in zip(self.train_feat_batch, self.train_label_batch):
                zlayer1 = self.forwardProp(weight_l1, feature, blayer1)
                alayer1 = self.sig(zlayer1)
                zlayer2 = self.forwardProp(weight_l2, alayer1, blayer2)
                alayer2 = self.softMax(zlayer2)

                delta3 = self.crossEntropy(alayer2, label)
                dweight_l2 = (alayer1.T).dot(delta3)
                dblayer2 = np.sum(delta3, axis = 0, keepdims = True)
                delta2 = delta3.dot(weight_l2.T)*self.gradSig(alayer1)
                dblayer1 = np.sum(delta2, axis = 0)
                dweight_l1= (feature.T).dot(delta2)

                weight_l1 -= self.learning_rate * dweight_l1
                weight_l2 -= self.learning_rate * dweight_l2
                blayer1 -= self.learning_rate * dblayer1
                blayer2 -= self.learning_rate * dblayer2

            # every 10th epoch
            if not (cur_epoch % 10):
                train_cost = self.trainCost(weight_l1, weight_l2, blayer1, blayer2)
                test_cost = self.testCost(weight_l1, weight_l2, blayer1, blayer2)
                self.train_accuracy.append(train_cost)
                self.test_accuracy.append(test_cost)
                print(f'Train Cost: {train_cost}\t\tTest Cost: {test_cost}\t\t\t\t', end='\r')

        self.model = {'W1': weight_l1, 
                       'B1': blayer1, 
                       'W2': weight_l2, 
                       'B2': blayer2}
          
    def dataLoader(self):
        """
        Load data and convert into mini batches.
        """
        self.train_data = pd.read_csv('../data/training_data.csv', header=None)
        self.test_data = pd.read_csv('../data/testing_data.csv', header=None)
        
        self.train_data_feats = self.train_data.iloc[:, :-3].to_numpy()
        self.train_data_labels = self.train_data.iloc[:, -3:].to_numpy()
        self.test_data_feats = self.test_data.iloc[:, :-3].to_numpy()
        self.test_data_labels = self.test_data.iloc[:, -3:].to_numpy()
        
        self.train_feat_batch = self.dfSplit(self.train_data_feats)
        self.train_label_batch = self.dfSplit(self.train_data_labels)
        self.test_feat_batch = self.dfSplit(self.test_data_feats)
        self.test_label_batch = self.dfSplit(self.test_data_labels)
    
    def softMax(self, z_mat):
        z_exp = np.exp(z_mat)
        interim_sum = np.sum(z_exp, axis=1, keepdims = True)
        return z_exp / interim_sum
    
    def crossEntropy(self, predicted, real):
        return (predicted-real) / real.shape[0]
    
    def preProcess(self):
        # Already done in preprocessing part and files saved.
        pass
    
    def forwardProp(self, weight_mat, left_out, bias):
        return left_out.dot(weight_mat) + bias
    
    def backwardProp(self, delt_r, weight_mat, act_right):
        return delt_r.dot(weight_mat.T) * (1 - np.power(act_right, 2))
       
    def trainCost(self, w1, w2, blayer1, blayer2):
        num = self.train_data_feats.shape[0]
        zlayer1 = self.forwardProp(w1, self.train_data_feats, blayer1)
        alayer1 = self.sig(zlayer1)
        zlayer2 = self.forwardProp(w2, alayer1, blayer2)
        alayer2 = self.softMax(zlayer2)

        iter_shape = alayer2.shape[0]

        for i in alayer2 :
            maxValInd = np.argmax(i)
            i[maxValInd] = 1
            i[(maxValInd+1)%self.num_neurons_out_layer] = 0
            i[(maxValInd+2)%self.num_neurons_out_layer] = 0

        count = 0
        for i in range(iter_shape):
            if np.array_equal(alayer2[i], self.train_data_labels[i]):
                count += 1            
        count *= 100
        return count/iter_shape
    
    def testCost(self, w1, w2, blayer1, blayer2):
        num = self.test_data_feats.shape[0]
        zlayer1 = self.forwardProp(w1, self.test_data_feats, blayer1)
        alayer1 = self.sig(zlayer1)
        zlayer2 = self.forwardProp(w2, alayer1, blayer2)
        alayer2 = self.softMax(zlayer2)

        iter_shape = alayer2.shape[0]

        for i in alayer2 :
            maxValInd = np.argmax(i)
            i[maxValInd] = 1
            i[(maxValInd+1)%self.num_neurons_out_layer] = 0
            i[(maxValInd+2)%self.num_neurons_out_layer] = 0

        count = 0
        for i in range(iter_shape):
            if np.array_equal(alayer2[i], self.test_data_labels[i]):
                count += 1            
        count *= 100
        return count/iter_shape
    
    def dfSplit(self, df): 
        df_list = []
        numChunks = len(df) // self.num_neurons_per_layer + 1
        for i in range(numChunks):
            df_list.append(df[i*self.num_neurons_per_layer:(i+1)*self.num_neurons_per_layer])
        return df_list
    
    @staticmethod
    def weightInitialiser(inpl1,inpl2):
        """
        Initialize values in dense layers all between -1 and 1.
        """
        rand_mat = np.random.rand(inpl1, inpl2)
        return (2 * rand_mat) - 1
    
    @staticmethod
    def gradSig(x):
        return (1/(1+np.exp(-x))) * (1 - (1/(1+np.exp(-x))))
    
    @staticmethod
    def sig(x):
        return (1/(1+np.exp(-x)))
    
nn1 = NeuralNetwork(
        num_hidden = 1,
        num_neurons_per_layer = 32,
        activation_func_hidden = 'sigmoid',
        num_neurons_out_layer=3,
        activation_func_output="softmax",
        opt_algo="SGD",
        loss_func="categorial_cross_entropy_loss",
        learning_rate=0.01,
        num_epochs=200)

nn1.fit()
nn1.plot()
