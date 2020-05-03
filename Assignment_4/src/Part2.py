#!/usr/bin/env python
# coding: utf-8
# @author: Shivam Kumar Jha
# @rollno: 17CS30033

from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

def accuracy(predicted, label):
	iter_shape = predicted.shape[0]
	count = 0
	for i in range(iter_shape):
		if np.array_equal(predicted[i], label[i]):
			count += 1            
	count *= 100
	return count/iter_shape

train_data = pd.read_csv('../data/training_data.csv', header=None)
test_data = pd.read_csv('../data/testing_data.csv', header=None)

train_data_feats = train_data.iloc[:, :-3].to_numpy()
train_data_labels = train_data.iloc[:, -3:].to_numpy()
test_data_feats = test_data.iloc[:, :-3].to_numpy()
test_data_labels = test_data.iloc[:, -3:].to_numpy()

nn1 = MLPClassifier(solver = 'sgd',
							activation = 'logistic',
							batch_size = 32,
                    		hidden_layer_sizes = (32),
							random_state = 1,
							learning_rate_init = 0.01,
							learning_rate = 'constant',
							max_iter = 200)

nn1.fit(train_data_feats, train_data_labels)
predicted_train_1a = nn1.predict(train_data_feats)
predicted_test_1a = nn1.predict(test_data_feats)
print("Part 2 Specification 1A :")
print("Train Accuracy = ", accuracy(predicted_train_1a, train_data_labels))
print("Test Accuracy = ", accuracy(predicted_test_1a, test_data_labels))

nn2 = MLPClassifier(solver = 'sgd',
					activation = 'relu',
					batch_size = 32,
                    hidden_layer_sizes = (64, 32), 
					random_state =1, 
					learning_rate_init = 0.01, 
					learning_rate = 'constant', 
					max_iter = 200)

nn2.fit(train_data_feats, train_data_labels)
predicted_train_1b = nn2.predict(train_data_feats)
predicted_test_1b = nn2.predict(test_data_feats)
print("\n\nPart 2 Specification 1B :")
print("Train Accuracy = ", accuracy(predicted_train_1b, train_data_labels))
print("Test Accuracy = ", accuracy(predicted_test_1b, test_data_labels))

