# Shivam Kumar Jha
# 17CS30033
# The functions are written in the order of the questions and solution to, for example 1a is named as a1_plot

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# loading training data
try:
    train_data = pd.read_csv('train.csv')
except FileNotFoundError:
    train_data = pd.read_csv(input('Enter path to training data: '))
    
# loading testing data
try:
    test_data = pd.read_csv('test.csv')
except FileNotFoundError:
    test_data = pd.read_csv(input('Enter path to test data: '))
    
def a1_plot():
    training = True
    for data in [train_data, test_data]:
        # print(train_data.describe())
        plt.scatter(data['Feature'], data[' Label'])
        plt.xlabel("Feature")
        plt.ylabel("Label")
        if training:
            plt.title("Training Data")
            training = False
        else:
            plt.title("Testing Data")
        plt.show()

a1_plot()