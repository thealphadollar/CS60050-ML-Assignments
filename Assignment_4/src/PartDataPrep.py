#!/usr/bin/env python
# coding: utf-8
# @author: Shivam Kumar Jha
# @rollno: 17CS30033

import pandas as pd
from sklearn.model_selection import train_test_split

with open('../data/seeds_dataset.txt', 'r') as data:
    with open('../data/seeds_dataset.csv', 'w+') as wData:
        for line in data.readlines():
            wData.write(','.join(line.split()))
            wData.write('\n')

df = pd.read_csv('../data/seeds_dataset.csv', header=None)
# df.head()

for i in range(7):
    df[i] = (df[i] - df[i].mean())/df[i].std(ddof=0)
# df.head()

one_hot = pd.get_dummies(df[7], prefix='_')
df = df.drop(7, axis=1)
df = df.join(one_hot)
# df.head()


train, test = train_test_split(df, test_size=0.2)

train.to_csv('../data/training_data.csv', header=None, index=False)
test.to_csv('../data/testing_data.csv', header=None, index=False)