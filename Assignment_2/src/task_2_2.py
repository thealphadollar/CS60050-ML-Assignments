import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from os import path

def g(inpu):
    return (1/(1+np.exp(-inpu)))

def cost(inpu, actual):
    return (actual*np.log10(inpu)) + ((1-actual)*(np.log10(1-inpu)))

# vectorize the function
g_vectorized = np.vectorize(g)
cost_vectorized = np.vectorize(cost)


def main():
    src_dir = path.join(path.dirname(path.realpath(__file__)), '..', 'data')
    results_dir = path.join(path.dirname(path.realpath(__file__)), '..', 'results')

    train_data = pd.read_csv(path.join(src_dir, 'dataset_A.csv'), sep=';')
    # print(train_data)
    train_data_label = train_data['quality'].to_numpy()
    train_data_feature = train_data.loc[:, train_data.columns != 'quality'].to_numpy()
    # print(train_data_feature.shape)

    learning_rate = 0.05

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
    print(f"Error on Data: {np.sum(cost(g(np.dot(X, coeff_vals)), train_data_label)) * (-1/m)}\n")

    with open(path.join(results_dir, 'parameters_sklearn_logistic.csv'), 'w') as f_out:
        f_out.write('intercept;')
        f_out.write(';'.join(list(train_data.loc[:, train_data.columns != 'quality'].columns.values)))
        f_out.write('\n')
        f_out.write(';'.join([str(x) for x in coeff_vals]))
    
    return coeff_vals

if __name__ == '__main__':
    main()
