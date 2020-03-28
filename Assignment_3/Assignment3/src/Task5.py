#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:13:09 2020

@author: praveshj
"""

#Calculating the NMI Score for all four clusterings

import math
import pandas as pd
import numpy as np


#X is a list of lists, where each list correponds to a class
#So to calculate entropy we just take the number in curr list and divide it by total number of elements, 
#after apply the formula on this it gives us the final entropy for list
def entropy_list(X):
    ele_val =[]
    num_elem = 0
    for i in X:
        ele_val.append(len(i))
        num_elem += len(i)
    entropy = 0
    for i in range(0, len(X)):
        entropy -= (ele_val[i]/num_elem)*math.log2(ele_val[i]/num_elem)
    return entropy


#To obtain the list of lists, in the format that we need to calculate entropy for each clustering
def get_list_from_file(file):
    fin_lis = []
    for i in range(0, 8):
        X = list(map(int, file.readline().split(',')))
        fin_lis.append(X)
    return fin_lis


#Conditional Entropy depends on the orginal class of the vector, and the class assigned to it
#So in order to get the conditional entropy, we take true label and assigned label as input
#what we did here is, according to the formula mentrion in the assignment. 
def get_conditional_entropy(label, cluster_list):
    entropy=0
    ele_val = []
    num_elem= 0;
    for i in cluster_list:
        num_elem += len(i)
        ele_val.append(len(i))
    for i in range(0, len(cluster_list)):
        tempE = 0
        curr_clust = cluster_list[i]
        for j in range(0, 8):
            countClass = 0
            if(len(curr_clust) == 0):
                print("Error Encountered")
                continue;
            for k in range(0, len(curr_clust)):
                if(label[curr_clust[k]] == j):
                    countClass+=1;
            if(countClass >0): tempE -= (countClass/len(curr_clust))*math.log2(countClass/len(curr_clust));
        entropy += (len(curr_clust)/num_elem)*tempE;
    return entropy
    

#The final formula for nmi is applied on three values obtained from above formula
def get_nmi(entropy_class, entropy_clust, conditional_entropy):
    return (2*conditional_entropy)/(entropy_class+entropy_clust)


#The Final NMI are printed for a file with this function
def print_nmi(file, label, entropy_class):
    clust_list = get_list_from_file(file);
    entropy_clust = entropy_list(clust_list)
    conditional_entropy = get_conditional_entropy(label, clust_list)
    fin_nmi = get_nmi(entropy_class, entropy_clust, conditional_entropy)
    print("NMI for the Given Clustering: ", fin_nmi)
    

datasetP = pd.read_csv('../data/AllBooks_baseline_DTM_Labelled.csv')
datasetP = datasetP.drop([13,13])

label = datasetP.iloc[:, 0]

#print(label)

#Label preprocessing from the initial dataset. Buddishm_CH12-> Buddish
label = np.asarray(label)
for i in range (0, len(label)):
    listName = label[i].split('_')
    label[i] = listName[0];
    
# print(label)

#Mapping the chapter to their numerical class
label[label == 'Buddhism'] = 0
label[label == 'TaoTeChing'] = 1
label[label == 'Upanishad'] = 2
label[label == 'YogaSutra'] = 3
label[label == 'BookOfProverb'] = 4
label[label == 'BookOfEcclesiastes'] = 5
label[label == 'BookOfEccleasiasticus'] = 6
label[label == 'BookOfWisdom'] = 7

#print(label)

class_list =[]
for i in range(0,8):
    temp = label[label == i];
    #print(temp)
    class_list.append(temp)

#print(class_list)
#Calculating the entropy for the given data, it's fixed and not dependent on any clustering
entropy_class = entropy_list(class_list)
# print(entropy_class)


#Final Calculation of NMI Scores for the files saved in earlier tasks
print('NMI SCORES:')
print('For Agglomerative Clustering')
file = open('../data/agglomerative.txt','r');
print_nmi(file, label, entropy_class)


print('For KMeans Clustering')
file = open('../data/kmeans.txt', 'r');
print_nmi(file, label, entropy_class)

print('For Agglomerative Clustering after PCA')
file = open('../data/agglomerative_reduced.txt', 'r');
print_nmi(file, label, entropy_class)


print('For KMeans After PCA')
file = open('../data/kmeans_reduced.txt', 'r');
print_nmi(file, label, entropy_class)

