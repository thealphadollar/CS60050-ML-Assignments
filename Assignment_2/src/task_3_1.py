import numpy as np
from os import path
from collections import defaultdict

class DTClassifier():
    def __init__(self, min_split=10):
        # value when to stop with minimum number of records
        self.min_split = min_split
        # class repeating the maximum number of times
        self.default_class = None
        self.tree = None

    @staticmethod
    def entropy(split_at):
        # method to calculate entropy
        # get different unique entries and their counts
        entries, nums = np.unique(split_at, return_counts = True)
        total = np.sum(nums)
        # calculate entropy with log2 formula for each of the separate entry
        return np.sum([((-nums[i])/total)*np.log2(nums[i]/total) for i in range(len(entries))])

    def info_gain(self, data, split_at, target_ind=11):
        # get vals and counts of unique different values for the particular column
        vals, nums = np.unique(data[:,split_at], return_counts=True)
        # get total sum of number of different values
        total_n = np.sum(nums)
        # calculate split entropy using weighted method with above entropy function
        split_entropy = np.sum([(nums[i]/total_n)*self.entropy(data[np.where(data[:,split_at]==vals[i])][:,target_ind]) for i in range(len(vals))])
        # return information gain which is current entropy - weighted entropy after split
        return self.entropy(data[:,target_ind]) - split_entropy

    def recur_fit(self, data, target_ind = 11, last_split=-1):
        # if only one class inside current subtree
        if len(np.unique(data[:,target_ind])) < 2:
            return np.unique(data[:,target_ind])[0]
        # if less than 10 members
        elif len(data) < self.min_split:
            return np.argmax(np.bincount(data[:,target_ind]))
        else:
            max_gain = -1e10
            split_at = None
            # calculate index at which split will give max info gain
            for val in range(11):
                if val != last_split:
                    info_gain = self.info_gain(data, val)
                    if  info_gain > max_gain:
                        max_gain = info_gain 
                        split_at = val
            # if split at chosen index will give only a single unique value, return max frequency class
            if len(np.unique(data[:,split_at])) == 1:
                return np.argmax(np.bincount(data[:,target_ind]))
            # create a subtree
            root = {'key': split_at}
            for diff_attr in np.unique(data[:,split_at]):
                # add subtrees to the current root with each split value
                root[diff_attr] = self.recur_fit(data[np.where(data[:,split_at]==diff_attr)], last_split=split_at)
            return root

    def fit(self, train_data_feature, train_data_label, target_ind=11):
        # concatenate labels to feature set to easier access with index
        data = np.c_[train_data_feature, train_data_label]
        # calculate default class which is class with maximum frequency
        self.default_class = np.argmax(np.bincount(data[:,target_ind]))
        # recursively create tree
        self.tree = self.recur_fit(data)

    def recur_predict(self, row, tree):
        # recursively find the class current row would belong to.
        try:
            val = tree[row[tree['key']]]
            # if the value obtained is a tree
            if isinstance(val, dict) == True:
                return self.recur_predict(row, val)
            # if value obtained is a class
            else:
                return val
        # in case none of the key or index matches
        except (KeyError, IndexError) as _:
            return self.default_class

    # function for predicting results
    def predict(self, data_feature):
        # create an numpy array with each of the decision tree prediction
        return np.array([self.recur_predict(data, self.tree) for data in data_feature])

    # method required for working with sklearn cross validate
    def get_params(self, deep = False):
        return {'min_split':self.min_split}
