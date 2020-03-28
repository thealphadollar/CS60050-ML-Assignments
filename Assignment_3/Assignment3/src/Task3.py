#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 01:37:42 2020

@author: praveshj
"""

import numpy as np
import math
import random

print("Data Loading. May Take Time....")

X = np.genfromtxt('../data/tfidf.csv',delimiter = ' ')

# print(X.shape)

        
#Cosine Similarity  and Distance will be same in each tast. There will be no difference
def cosine_similarity(X, Y):
    dotP = np.dot(X, Y);
    modX = math.sqrt(X.dot(X))
    modY= math.sqrt(Y.dot(Y))
    return dotP/(modX*modY)


def dist(X, Y):
    return math.exp(-1*cosine_similarity(X, Y));



#The functions returns the index of centroid which is at the minimum distance from a the Vector X
def min_dist(centroids, X):
    dis_min = 1e10
    index = 0
    for i in range(0, 8):
        temp = dist(centroids[i], X)
        if dis_min > temp:
            dis_min = temp;
            index = i;
    return index;

num_doc = X.shape[0]
dist_mat = np.zeros((X.shape[0],X.shape[0]))


#Distance Matrix, so we don't have to calculate distance again and again, between known vectors
for i in range(0, num_doc):
    for j in range(0, num_doc):
        if i == j :
            dist_mat[i][j] = 1e10
        else:
            dist_mat[i][j] = dist(X[i], X[j])
            
#clusterClass stores the cluster assigned to each document by kmeans
clusterClass = np.zeros((num_doc, 1))
#Centroids is the array of the 8 cluster centroids
centroids = np.zeros((8, X.shape[1]))
for i in range(0, 8):
    centroids[i]= X[np.random.random_integers(0, X.shape[0])]
    
    
#For each iteration we will find the mean of the all the elements, assigned to that class,
# the new centroid will assigned
# based on it    
for j in range(0, 1000):
    print('Iteration', j)
    for i in range(0, num_doc):
        clusterClass[i] = min_dist(centroids, X[i])
    
    for thisClass in range(0, 8):
        temp = np.zeros((1, X.shape[1]))
        count = 0
        for i in range(0, num_doc):
            if(clusterClass[i] == thisClass):
                temp = temp + X[i];
                count +=1
        centroids[thisClass] = temp/count
    
#The final output file is saved in kmeans.txt in the format asked
file= open('../data/kmeans.txt', 'w')
for thisClass in range(0, 8):
    temp = np.zeros((1, X.shape[1]))
    count = 0
    for i in range(0, num_doc):
        if(clusterClass[i] == thisClass):
            if(count == 0):
                file.write(str(i));
                count = 1
            else: file.write(',' + str(i))
    file.write('\n')
file.close()


