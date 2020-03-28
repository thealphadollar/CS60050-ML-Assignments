#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 22:11:58 2020

@author: praveshj
"""

import numpy as np
import math

#Loading the data from saved text file. 
print("Data Loading. May Take Time....")
X = np.genfromtxt('../data/tfidf.csv',delimiter = ' ')



#Defining a class cluster, which store all the vectors in the cluster, and their indices in the dataset
class cluster:
    def __init__(self, vectors, index):
        self.vectors = vectors
        self.index= index
        
        


#function to cosine similarity between two vectors X and Y
def cosine_similarity(X, Y):
    dotP = np.dot(X, Y);
    modX = math.sqrt(X.dot(X))
    modY= math.sqrt(Y.dot(Y))
    return dotP/(modX*modY)


#Calculating the Dist using cosine similarity for two vectors
def dist(X, Y):
    return math.exp(-1*cosine_similarity(X, Y));


#Using Single Linkage strategy that is, min distance between any two points in the cluster as the distance between clusters
#To calculate the distance between clusters
def clusterDist( a, b, dist_mat):
    min_dis = 1e10
    for i in range(0, len(a.index)):
        for j in range(0, len(b.index)):
            if min_dis > dist_mat[a.index[i]][b.index[j]]:
                min_dis = dist_mat[a.index[i]][b.index[j]]
    return min_dis;


#Function to merge two clusters, by creating a new cluster, returning it. 
def mergeCluster(a, b):
    vectors = []
    index = []
    vectors.extend(a.vectors);
    index.extend(a.index);
    vectors.extend(b.vectors);
    index.extend(b.index);
    return cluster(vectors, index);
    
#All the existing clusters will be stored in the list_cluster[] list
list_cluster = []
num_cluster = len(list_cluster) 


for i in range (0, X.shape[0]):
    list_cluster.append(cluster([X[i]], [i] ))

#Always contains the current of elements in the cluster
num_cluster = len(list_cluster)
# print(num_cluster)


#Stores the distance between any two vectors, so that it's not calculated many times 
#Helps us save unnecessary calculation
dist_mat = np.zeros((num_cluster, num_cluster))

for i in range(0, num_cluster):
    for j in range(0, num_cluster):
        if i == j :
            dist_mat[i][j] = 1e10
        else:
            dist_mat[i][j] = dist(X[i], X[j])



#At the end we need 8 clusters, so we will continue to merge, the clusters with minimum distance
while(num_cluster > 8):
    min_dis=1e10
    X = 0
    Y = 0
    for i in range(0, num_cluster):
        for j in range(i+1 , num_cluster):
            temp= clusterDist(list_cluster[i], list_cluster[j], dist_mat)
            #print("temp = ", temp)
            #print("min_dist = ", min_dis)
            if( min_dis > temp):
                min_dis = temp
                X = list_cluster[i]
                Y = list_cluster[j]
        #print('Done')
    print("Clusters Remaing: ",num_cluster)
    list_cluster.remove(X)
    list_cluster.remove(Y)
    list_cluster.append(mergeCluster(X, Y))
    num_cluster = num_cluster-1;


#Saving the final Result in asked format
   
file = open('../data/agglomerative.txt', 'w')
for clust in list_cluster:
    clust.index = sorted(clust.index)
    for i in range(0, len(clust.index)):
        if(i != len(clust.index ) -1 ):
            file.write(str(clust.index[i])+',')
        else:
            file.write(str(clust.index[i]) + '\n')
            
file.close()