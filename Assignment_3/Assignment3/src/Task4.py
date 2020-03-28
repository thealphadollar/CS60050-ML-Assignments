#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 03:52:50 2020

@author: praveshj
"""

from sklearn.decomposition import PCA
import numpy as np
import math

print("Data Loading. May Take Time....")
X = np.genfromtxt('../data/tfidf.csv',delimiter = ' ')

#PCA using sklearn, components reduced to 100 form 8266
pca = PCA(n_components = 100)

#Reducting the Principal Components to 100
X = pca.fit_transform(X)

#print(X.shape)

class cluster:
    def __init__(self, vectors, index):
        self.vectors = vectors
        self.index= index
        
        

#Cosine Similarity is defined in a similar manner as all the files
def cosine_similarity(X, Y):
    dotP = np.dot(X, Y);
    modX = math.sqrt(X.dot(X))
    modY= math.sqrt(Y.dot(Y))
    return dotP/(modX*modY)


def dist(X, Y):
    return math.exp(-1*cosine_similarity(X, Y));


#Since we are doing both,  Agglomerative as well as KMeans, functions for both of them are present

#Agglomerative Clustering
def clusterDist( a, b, dist_mat):
    min_dis = 1e10
    for i in range(0, len(a.index)):
        for j in range(0, len(b.index)):
            if min_dis > dist_mat[a.index[i]][b.index[j]]:
                min_dis = dist_mat[a.index[i]][b.index[j]]
    return min_dis;

def mergeCluster(a, b):
    vectors = []
    index = []
    vectors.extend(a.vectors);
    index.extend(a.index);
    vectors.extend(b.vectors);
    index.extend(b.index);
    return cluster(vectors, index);
    
list_cluster = []
num_cluster = len(list_cluster)

for i in range (0, X.shape[0]):
    list_cluster.append(cluster([X[i]], [i] ))

num_cluster = len(list_cluster)
# print(num_cluster)

dist_mat = np.zeros((num_cluster, num_cluster))

for i in range(0, num_cluster):
    for j in range(0, num_cluster):
        if i == j :
            dist_mat[i][j] = 1e10
        else:
            dist_mat[i][j] = dist(X[i], X[j])

while(num_cluster > 8):
    min_dis=1e10
    a = 0
    b = 0
    for i in range(0, num_cluster):
        for j in range(i+1 , num_cluster):
            temp= clusterDist(list_cluster[i], list_cluster[j], dist_mat)
            #print("temp = ", temp)
            #print("min_dist = ", min_dis)
            if( min_dis > temp):
                min_dis = temp
                a = list_cluster[i]
                b = list_cluster[j]
        #print('Done')
    print('Clusters Remaining', num_cluster)
    list_cluster.remove(a)
    list_cluster.remove(b)
    list_cluster.append(mergeCluster(a, b))
    num_cluster = num_cluster-1;
   
file = open('../data/agglomerative_reduced.txt', 'w')
for clust in list_cluster:
    for i in range(0, len(clust.index)):
        if(i != len(clust.index ) -1 ):
            file.write(str(clust.index[i])+',')
        else:
            file.write(str(clust.index[i]) + '\n')
            
file.close()


#Agglomerative Clustering Done
#KMeans Clustering Started for the PCA reduced dataset
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
            
clusterClass = np.zeros((num_doc, 1))
centroids = np.zeros((8, X.shape[1]));
for i in range(0, 8):
    centroids[i]= X[np.random.random_integers(0, X.shape[0])]
    
    
    
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
    
file= open('../data/kmeans_reduced.txt', 'w')
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

#KMeans reduced clustering dones


