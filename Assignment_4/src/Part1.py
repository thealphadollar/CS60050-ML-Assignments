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
        if activation_func_hidden == 'sigmoid':
            self.activation_func_output = self.sig
            self.derivative_act_func_output = self.gradSig
        elif activation_func_hidden == 'relu':
            self.activation_func_output = self.relu
            self.derivative_act_func_output = self.reluDerivative
        self.num_neurons_out_layer = num_neurons_out_layer
        self.opt_algo = opt_algo
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
    
    def fit(self):
        self.dataLoader()
        self.train()

    def showAccu(self, part):
        print(f"Part 1 Specification {part} :")
        print("Train Accuracy = ", self.train_accuracy[-1])
        print("Test Accuracy = ", self.test_accuracy[-1])
    
    def plot(self, part):
        iterations = [x for x in range(0,200,10)]
        plt.plot(iterations, self.train_accuracy[1:], label="Train")
        plt.plot(iterations, self.test_accuracy[1:], label="Test")
        plt.xlabel('Cost')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f"Part {part} Accuracy Vs Iterations")
        plt.show()
        plt.clf()
        
    def train(self):
        self.test_accuracy = list()
        self.train_accuracy = list()

        layer_weights = self.weightInitialiser(7, self.num_neurons_per_layer, self.num_neurons_out_layer)
        layer_bias = self.biasInitialiser(self.num_neurons_per_layer, self.num_neurons_out_layer)

        train_prediction, test_prediction = self.predict(layer_weights, layer_bias)
        self.train_accuracy.append(self.curAccu(train_prediction, self.train_data_labels))
        self.test_accuracy.append(self.curAccu(test_prediction, self.test_data_labels))

        for cur_epoch in range(self.num_epochs):
            
            for feat_batch, label_batch in zip(self.train_feat_batch, self.train_label_batch):
                act_right = self.forwardProp(feat_batch, layer_weights, layer_bias)
                layer_weights, layer_bias = self.backwardProp(layer_weights, layer_bias, act_right, label_batch)

            # every 10th epoch
            if not (cur_epoch % 10):
                train_prediction, test_prediction = self.predict(layer_weights, layer_bias)
                self.train_accuracy.append(self.curAccu(train_prediction, self.train_data_labels))
                self.test_accuracy.append(self.curAccu(test_prediction, self.test_data_labels))


        self.model_ = {"layer_weights": layer_weights,
                       "layer_bias": layer_bias}
          
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

    def predict(self, layer_weights, layer_bias):
        prediction_train = self.forwardProp(self.train_data_feats, layer_weights, layer_bias)
        shape = prediction_train[-1].shape
        pred_train2 = np.zeros(shape)
        pred_train2[np.arange(shape[0]), np.argmax(prediction_train[-1], axis=-1)] = 1

        prediction_test = self.forwardProp(self.test_data_feats, layer_weights, layer_bias)
        pred_test2 = np.zeros(prediction_test[-1].shape)
        pred_test2[np.arange(prediction_test[-1].shape[0]), np.argmax(prediction_test[-1], axis=-1)] = 1
        return pred_train2, pred_test2

    def forwardProp(self, features, layer_weights, layer_bias):
        feat_list = list()
        feat_list.append(features)
        for layer_weight, layer_bia in zip(layer_weights[:-1], layer_bias[:-1]):
            feat_list.append(self.activation_func_output((feat_list[-1]).dot(layer_weight) + layer_bia))
        feat_list.append(self.softMax((feat_list[-1]).dot(layer_weights[-1]) + layer_bias[-1]))
        return feat_list
    
    def backwardProp(self, layer_weights, layer_bias, act_right, label):
        delta = act_right[-1]
        delta[label==1] -= 1
        
        for i in range(len(layer_weights)-1, 0, -1):
            delta_next = delta.dot(layer_weights[i].T)*self.derivative_act_func_output(act_right[i])
            layer_weights[i] = layer_weights[i] - self.learning_rate * ((act_right[i].T).dot(delta))
            layer_bias[i] = layer_bias[i] - self.learning_rate * np.sum(delta, axis = 0, keepdims = True)
            delta = delta_next
        return layer_weights, layer_bias
       
    # def trainCost(self, w1, w2, blayer1, blayer2):
    #     num = self.train_data_feats.shape[0]
    #     zlayer1 = self.forwardProp(w1, self.train_data_feats, blayer1)
    #     alayer1 = self.activation_func_output(zlayer1)
    #     zlayer2 = self.forwardProp(w2, alayer1, blayer2)
    #     alayer2 = self.softMax(zlayer2)

    #     iter_shape = alayer2.shape[0]

    #     for i in alayer2 :
    #         maxValInd = np.argmax(i)
    #         i[maxValInd] = 1
    #         i[(maxValInd+1)%self.num_neurons_out_layer] = 0
    #         i[(maxValInd+2)%self.num_neurons_out_layer] = 0

    #     count = 0
    #     for i in range(iter_shape):
    #         if np.array_equal(alayer2[i], self.train_data_labels[i]):
    #             count += 1            
    #     count *= 100
    #     return count/iter_shape

    # def testCost(self, w1, w2, blayer1, blayer2):
    #     num = self.test_data_feats.shape[0]
    #     zlayer1 = self.forwardProp(w1, self.test_data_feats, blayer1)
    #     alayer1 = self.activation_func_output(zlayer1)
    #     zlayer2 = self.forwardProp(w2, alayer1, blayer2)
    #     alayer2 = self.softMax(zlayer2)

    #     iter_shape = alayer2.shape[0]

    #     for i in alayer2 :
    #         maxValInd = np.argmax(i)
    #         i[maxValInd] = 1
    #         i[(maxValInd+1)%self.num_neurons_out_layer] = 0
    #         i[(maxValInd+2)%self.num_neurons_out_layer] = 0

    #     count = 0
    #     for i in range(iter_shape):
    #         if np.array_equal(alayer2[i], self.test_data_labels[i]):
    #             count += 1            
    #     count *= 100
    #     return count/iter_shape
    
    @staticmethod
    def curAccu(predicted, orig):
        iter_shape = predicted.shape[0]
        count = 0
        for i in range(iter_shape):
            if np.array_equal(predicted[i], orig[i]):
                count += 1            
        count *= 100
        return count/iter_shape

    @staticmethod
    def dfSplit(df): 
        df_list = []
        numChunks = len(df) // 32 + 1
        for i in range(numChunks):
            df_list.append(df[i*32:(i+1)*32])
        return df_list
    
    @staticmethod
    def weightInitialiser(data_dim, layers, num_out_layers):
        """
        Initialize values in dense layers all between -1 and 1.
        """
        weights = list()
        for layer in range(len(layers)):
            if not layer:
                rand_mat = np.random.rand(data_dim, layers[layer])
            else:
                rand_mat = np.random.rand(layers[layer-1], layers[layer])
            weights.append((2 * rand_mat) - 1)
        rand_mat = np.random.rand(layers[-1], num_out_layers)
        weights.append((2 * rand_mat) - 1)
            
        return weights

    @staticmethod
    def biasInitialiser(layers, num_out_layers):
        """
        Initialize values in dense layers all between -1 and 1.
        """
        bias = list()
        for layer in range(len(layers)):
            bias.append(np.zeros((1, layers[layer])))
        bias.append(np.zeros((1, num_out_layers)))
            
        return bias
    
    @staticmethod
    def gradSig(x):
        return (1/(1+np.exp(-x))) * (1 - (1/(1+np.exp(-x))))
    
    @staticmethod
    def sig(x):
        return (1/(1+np.exp(-x)))

    @staticmethod
    def reluDerivative(x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    
    @staticmethod
    def relu(x):
        x[x<=0] = 0
        return x
    
nn1 = NeuralNetwork(
        num_hidden = 1,
        num_neurons_per_layer = (32, ),
        activation_func_hidden = 'sigmoid',
        num_neurons_out_layer=3,
        activation_func_output="softmax",
        opt_algo="SGD",
        loss_func="categorial_cross_entropy_loss",
        learning_rate=0.01,
        num_epochs=200)

nn1.fit()
nn1.showAccu("1A")
nn1.plot("1A")

print('\n')

nn2 = NeuralNetwork(
        num_hidden = 2,
        num_neurons_per_layer = (64, 32),
        activation_func_hidden = 'relu',
        num_neurons_out_layer=3,
        activation_func_output="softmax",
        opt_algo="SGD",
        loss_func="categorial_cross_entropy_loss",
        learning_rate=0.01,
        num_epochs=200)

nn2.fit()
nn2.showAccu("1B")
nn2.plot("1B")