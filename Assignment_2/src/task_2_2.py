import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from os import path

from task_2_1 import g_vectorized, cost_vectorized

def main():
    src_dir = path.join(path.dirname(path.realpath(__file__)), '..', 'data')

    train_data = pd.read_csv(path.join(src_dir, 'dataset_A.csv'), sep=';')
    # print(train_data)
    train_data_label = train_data['quality'].to_numpy()
    train_data_feature = train_data.loc[:, train_data.columns != 'quality'].to_numpy()
    # print(train_data_feature.shape)

    logregressor = LogisticRegression(penalty='none', solver='saga')
    logregressor.fit(train_data_feature, train_data_label)
    
    # setting initial value of the parameters as 1
    coeff_vals = [list(logregressor.intercept_)[0]]
    coeff_vals.extend(list(logregressor.coef_)[0])
    
    print(f"Parameters: {coeff_vals}\t\t")
    # creating numpy vector with feature data
    m,n = train_data_feature.shape
    X = np.zeros(shape=(m,n+1))
    for i in range(m):
        for j in range(n+1):
            if j==0:
                X[i][j] = 1
            else:
                X[i][j] = train_data_feature[i][j-1]
    print(f"Error on Data: {np.sum(cost_vectorized(g_vectorized(np.dot(X, coeff_vals)), train_data_label)) * (-1/m)}\n")

if __name__ == '__main__':
    main()
