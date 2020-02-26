import numpy as np
import pandas as pd
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

    m,n = train_data_feature.shape

    # creating numpy vector with feature data
    X = np.zeros(shape=(m,n+1))
    for i in range(m):
        for j in range(n+1):
            if j==0:
                X[i][j] = 1
            else:
                X[i][j] = train_data_feature[i][j-1]
    # setting initial value of the parameters as 1
    coeff_vals = np.ones(n+1)
    # stores previous cost
    prev_jtheta = None
    # calculate loss
    loss = (g(np.dot(X, coeff_vals)) - train_data_label)
    # current cost
    cur_jtheta = np.sum(cost(g(np.dot(X, coeff_vals)), train_data_label)) * (-1/m)
    # setting convergence when the difference between consecutive error is less than 0.0000001
    while (prev_jtheta is None or abs(prev_jtheta - cur_jtheta) > 0.0000001):
        # gradient descent with vector notation, simultaneous calculation
        descent_vals = np.dot(X.transpose(), loss) * (learning_rate / m)
        # update all coefficients with descent
        coeff_vals = coeff_vals - descent_vals
        prev_jtheta = cur_jtheta
        # calculate new cost
        loss = (g(np.dot(X, coeff_vals)) - train_data_label)
        cur_jtheta = np.sum(cost(g(np.dot(X, coeff_vals)), train_data_label)) * (-1/m)
        print(f"Difference between consecutive costs: {abs(prev_jtheta - cur_jtheta)}\t", end="\r", flush=True)
    print(f"Parameters: {coeff_vals}\t\t")
    print(f"Error on Data: {cur_jtheta}\n")

    with open(path.join(results_dir, 'parameters_manual_logistic.csv'), 'w') as f_out:
        f_out.write(';'.join(list(train_data.loc[:, train_data.columns != 'quality'].columns.values)))
        f_out.write('\n')
        f_out.write(';'.join([str(x) for x in coeff_vals]))
    
    return coeff_vals

if __name__ == '__main__':
    main()
