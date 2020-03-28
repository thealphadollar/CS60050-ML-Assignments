import pandas as pd
import numpy as np
import math

#Reading Primary Data
datasetP = pd.read_csv('../data/AllBooks_baseline_DTM_Labelled.csv')

# print(datasetP.shape)

#Dropping the Empty Row from data
datasetP = datasetP.drop([13,13])
# print(datasetP.shape)


#print(datasetP.head())

dftMat = np.asarray(datasetP)

#Separating the Labels 
label = dftMat[:, 0]
#print(label)


#Separating the DFT Matrix
X = dftMat[:, 1:]


#Removing the Term after _ Buddishm_Ch14 --> Buddishm
for i in range (0, len(label)):
    listName = label[i].split('_')
    label[i] = listName[0];
    
#print(label)
# print(X.shape)

df = np.zeros(( X.shape[1]))


#Calculating the df, Document Frequecy for Each term -> Number of Documents in which the term is present
#If Frequency >0 it means the term is present in the document, and thus will in the np array obtain by the boolean condition

# print(df.shape)
for i in range (0 , X.shape[1]):
    temp= X[:, i]    
    df[i] = temp[temp > 0].shape[0]

print("DF Calculated")
#df = np.reshape(df, (1, len(df)))
#print(df)




#Storing the value of the formula in a temporary variable
temp = (1 + X.shape[0])/(1 + df)

for i in range(0, len(temp)):
    temp[i] = math.log(temp[i])


# print(temp.shape)

idf = temp;
tfidf = np.zeros((X.shape))

#Final TDFIF matrix calcualtion by the given formula
for i in range(0, X.shape[0]):
    #print(X[i])
    tfidf[i] = X[i]*idf
    
count = 0

print("TF-IDF Calculated")

vec_len = np.zeros((X.shape[0]))


#Calculating the vector length for all the vectors
for i in range (0, X.shape[0]):
    vec_len[i] = math.sqrt(tfidf[i].dot(tfidf[i]))
    
    
#print(vec_len == 0)


#Normalisation done for the final Array. Diving by the vector length -> that is sqrt of sum of squares



for i in range(0, X.shape[0]):
    tfidf[i] = tfidf[i]/vec_len[i] 
print("Normalised TFIDF Calculated")

#Saving the tfdif matrix as a csv for the next assignment
np.savetxt('../data/tfidf.csv', tfidf)



