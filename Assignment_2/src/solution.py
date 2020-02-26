# Shivam Kumar Jha
# 17CS30033
# The functions are written in the order of the questions and solution to, for example 1a is named as _1a_plot

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from copy import copy
import json

# loading training data
try:
    train_data = pd.read_csv('train.csv')
except FileNotFoundError:
    train_data = pd.read_csv(input('Enter path to training data: '))
train_data_feature = train_data['Feature'].to_numpy()
train_data_label = train_data[' Label'].to_numpy()

# loading testing data
try:
    test_data = pd.read_csv('test.csv')
except FileNotFoundError:
    test_data = pd.read_csv(input('Enter path to test data: '))    
test_data_feature = test_data['Feature'].to_numpy()
test_data_label = test_data[' Label'].to_numpy()

# global variable to store training error, test error and predicted parameters
# index 0 corresponds to polynomial of degree 1 and so on
all_train_error = []
all_test_error = []
predicted_parameters = []
# highest degree polynomial to build starting from 0
till_n = 9
learning_rate = 0.05
# size of the training data
m = train_data_feature.size

# ======================
# Helper Functions Below
# ======================

# calculates polynomial based on given values and coefficients
def poly_calc(x, coeffs):
    o = len(coeffs)
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y

# taking variables, returns the cost
def cost_function(X, Y, m, coeffs):
    # calculates sigma of polynomial values over all X minus Y
    sum_ = sum([(poly_calc(X.iloc[ind], coeffs)-Y.iloc[ind])**2 for ind in range(m)])
    return sum_ / (2*m)

# returns best, worst curve based on training data error
def get_extreme_curves():
    # loading data from json files
    with open("test_error.json", 'r') as f:
        all_test_error = json.load(f)
    with open("train_error.json", 'r') as f:
        all_train_error = json.load(f)
    with open("parameters.json", 'r') as f:
        predicted_parameters = json.load(f)
    best_train_curve = None
    worst_train_curve = None
    min_train_error = min(all_train_error)
    max_train_error = max(all_train_error)
    for index, train_error in enumerate(all_train_error):
        if train_error == min_train_error:
            best_train_curve = predicted_parameters[index]
        if train_error == max_train_error:
            worst_train_curve = predicted_parameters[index]
    return np.array(best_train_curve), np.array(worst_train_curve)

# =========================
# Solutions Functions Below
# =========================

# plotting data, answer to 1a
def _1a_plot():
    training = True
    for data in [train_data, test_data]:
        # print(train_data.describe())
        plt.scatter(data['Feature'], data[' Label'])
        plt.xlabel("Feature")
        plt.ylabel("Label")
        if training:
            plt.title("Training Data")
            training = False
        else:
            plt.title("Testing Data")
        plt.show()

# fitting curve with learning rate 0.05, answer to 1b
def _1b_fitting():
    # varying n from 1 to 9
    for n in range(1, till_n+1):
        print(f"=========== Polynomial Degree {n} ===========")
        # creating numpy vector with feature data
        X = np.zeros(shape=(m,n+1))
        for i in range(m):
            for j in range(n+1):
                X[i][j] = np.power(train_data_feature[i], j)
        # setting initial value of the parameters as 1
        coeff_vals = np.ones(n+1)
        # stores previous cost
        prev_jtheta = None
        # calculate loss
        loss = np.dot(X, coeff_vals) - train_data_label
        # current cost
        cur_jtheta = np.sum(loss ** 2) / (2 * m)
        # setting convergence when the difference between consecutive error is less than 0.00000001
        while (prev_jtheta is None or abs(prev_jtheta - cur_jtheta) > 0.00000001):
            # gradient descent with vector notation, simultaneous calculation
            descent_vals = np.dot(X.transpose(), loss) * (learning_rate / m)
            # update all coefficients with descent
            coeff_vals = coeff_vals - descent_vals
            prev_jtheta = cur_jtheta
            # calculate new cost
            loss = np.dot(X, coeff_vals) - train_data_label
            cur_jtheta = np.sum(loss ** 2) / (2 * m)
            print(f"Difference between consecutive costs: {abs(prev_jtheta - cur_jtheta)}\t", end="\r", flush=True)
        predicted_parameters.append(coeff_vals.tolist())
        all_train_error.append(cur_jtheta.tolist())
        test_error = cost_function(test_data['Feature'], test_data[' Label'], len(test_data.index), coeff_vals)
        all_test_error.append(test_error)
        print(f"Parameters: {coeff_vals}\t\t")
        print(f"Squared Error on Test Data: {test_error}\n")

        # generating predicted values and saving as predicted_labels_n.csv where n is the degree of polynomial
        data = [[x, poly_calc(x, coeff_vals)] for x in test_data['Feature']]
        predicted_labels = pd.DataFrame(data, columns=["Feature", "Label"])
        predicted_labels.to_csv(f'predicted_labels_{n}.csv', index=False)
    # storing all predicted polynomials in "predicted_parameters.txt"
    with open("predicted_parameters.txt", "w+") as f:
        for index, parameter in enumerate(predicted_parameters):
            f.write(f"Predicted Parameters for Degree {index+1}: {parameter}\n")
    # saving parameters to load in later functions
    with open("parameters.json", 'w+') as f:
        json.dump(predicted_parameters, f)
    # saving train errors to load in later functions
    with open("train_error.json", 'w+') as f:
        json.dump(all_train_error, f)
    # saving test errors to load in later functions
    with open("test_error.json", 'w+') as f:
        json.dump(all_test_error, f)
        

# plotting the predicted polynomials, answer to 2a
def _2a_plots():
    # loading data from json file
    with open("parameters.json", 'r') as f:
        predicted_parameters = json.load(f)
    for index, coeffs in enumerate(predicted_parameters):
        plot_on = np.linspace(train_data['Feature'].min(), train_data['Feature'].max())
        plt.plot(plot_on, poly_calc(plot_on, coeffs))
        plt.xlabel("Feature")
        plt.ylabel("Label")
        plt.title(f"Polynomial of Degree {index+1}")
        # print(f"Plot for polynomial with coefficients {coeffs}")
        plt.show()

# plotting the squared error for training and test data for the values of n
def _2b_plots():
    # loading data from json files
    with open("test_error.json", 'r') as f:
        all_test_error = json.load(f)
    with open("train_error.json", 'r') as f:
        all_train_error = json.load(f)
    labels = [str(x) for x in range(1,till_n+1)]

    x = np.arange(len(labels))  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, all_train_error, width, label='Train Error')
    rects2 = ax.bar(x + width/2, all_test_error, width, label='Test Error')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Error')
    ax.set_xlabel('n -->')
    ax.set_title('Train and Test Errors For Varying n')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Attach a text label above each bar in *rects*, displaying its height
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{0:.7f}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()

# solution to part 3a, lasso regularisation
# the function is very similar to 2b fitting except that we do not start from all 1s
# and used the improvised cost function
def _3a_lasso():
    best_curve, worst_curve = get_extreme_curves()
    result_curves = []
    lasso_training_errors = []
    lasso_test_errors = []
    is_best = True
    for curve in [best_curve, worst_curve]:
        if is_best:
            curve_type = "Best Curve"
            is_best = False
        else:
            curve_type = "Worst Curve"
        n = len(curve) - 1
        # create numpy vector from feature data
        X = np.zeros(shape=(m,n+1))
        for i in range(m):
            for j in range(n+1):
                X[i][j] = np.power(train_data_feature[i], j)
        curves = []
        train_errors = []
        test_errors = []
        for lambd in [.25, .5, .75, 1]:
            print(f"=========== Lasso Regularisation {curve_type}: Polynomial Degree {n} Lambda {lambd} ===========")
            print(f"Polynomial is: {curve}")
            coeff_vals = copy(curve)
            prev_jtheta = None
            loss = np.dot(X, coeff_vals) - train_data_label
            cur_jtheta = (np.sum(loss ** 2) + (lambd * sum([abs(x) for x in coeff_vals])))/ (2 * m)
            # setting convergence when the difference between consecutive error is less than 0.00000001
            while (prev_jtheta is None or abs(prev_jtheta - cur_jtheta) > 0.00000001):
                descent_vals = np.dot(X.transpose(), loss) * (learning_rate / m)
                # using regularised descent values, saving zeroth term to put back as in later on.
                initial_coeff = coeff_vals[0]
                coeff_vals = (coeff_vals * (1-learning_rate *(lambd/m))) - descent_vals
                coeff_vals[0] = initial_coeff - descent_vals[0]
                prev_jtheta = cur_jtheta
                loss = np.dot(X, coeff_vals) - train_data_label
                cur_jtheta = (np.sum(loss ** 2) + (lambd * sum([abs(x) for x in coeff_vals])))/ (2 * m)
                print(f"Difference between consecutive costs: {abs(prev_jtheta - cur_jtheta)}\t", end="\r", flush=True)
            curves.append(coeff_vals)
            test_error = cost_function(test_data['Feature'], test_data[' Label'], len(test_data.index), coeff_vals) + ((lambd * sum([abs(x) for x in coeff_vals])) / (2 * len(test_data.index)))
            # storing squared (not lasso) train and test errors for later use in plotting
            train_errors.append(cur_jtheta - ((lambd * sum([abs(x) for x in coeff_vals])) / (2 * m)))
            test_errors.append(test_error - ((lambd * sum([abs(x) for x in coeff_vals])) / (2 * len(test_data.index))))
            print(f"New Parameters: {coeff_vals}\t\t")
            print(f"Lasso Error on Test Data: {test_error}")
            print(f"Squared Error on Test Data: {test_error - ((lambd * sum([abs(x) for x in coeff_vals])) / (2 * len(test_data.index)))}\n")
        result_curves.append(curves)
        lasso_training_errors.append(train_errors)
        lasso_test_errors.append(test_errors)
    
    # creating the graphs for all lambda for each of the two curves
    for i in range(2):
        # plotting test and train error for varying lambda
        labels = [0.25, 0.5, 0.75, 1]

        x = np.arange(len(labels))  # the label locations
        width = 0.5  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, lasso_training_errors[i], width, label='Lasso Train Error')
        rects2 = ax.bar(x + width/2, lasso_test_errors[i], width, label='Lasso Test Error')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Error')
        ax.set_xlabel('lambda -->')
        if i == 0:
            title = 'Best Curve: Train and Test Errors For Varying Lambda With Lasso'
        if i == 1:
            title = 'Worst Curve: Train and Test Errors For Varying Lambda With Lasso'
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        # display height of reactange
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{0:.7f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        fig.tight_layout()
        plt.show()

# solution to part 3b, ridge regularisation
# the function is very similar to 2b fitting except that we do not start from all 1s
# and used the improvised cost function
def _3b_ridge():
    best_curve, worst_curve = get_extreme_curves()
    result_curves = []
    ridge_training_errors = []
    ridge_test_errors = []
    is_best = True
    for curve in [best_curve, worst_curve]:
        n = len(curve) - 1
        X = np.zeros(shape=(m,n+1))
        for i in range(m):
            for j in range(n+1):
                X[i][j] = np.power(train_data_feature[i], j)
        if is_best:
            curve_type = "Best Curve"
            is_best = False
        else:
            curve_type = "Worst Curve"
        curves = []
        train_errors = []
        test_errors = []
        for lambd in [.25, .5, .75, 1]:
            print(f"=========== Ridge Regularisation {curve_type}: Polynomial Degree {n} Lambda {lambd} ===========")
            print(f"Polynomial is: {curve}")
            coeff_vals = copy(curve)
            prev_jtheta = None
            loss = np.dot(X, coeff_vals) - train_data_label
            cur_jtheta = (np.sum(loss ** 2) + (lambd * sum([x**2 for x in coeff_vals])))/ (2 * m)
            # setting convergence when the difference between consecutive error is less than 0.00000001
            while (prev_jtheta is None or abs(prev_jtheta - cur_jtheta) > 0.00000001):
                descent_vals = np.dot(X.transpose(), loss) * (learning_rate / m)
                # using regularised descent values, saving zeroth term to put back as in later on.
                initial_coeff = coeff_vals[0]
                coeff_vals = (coeff_vals * (1-learning_rate * (lambd/m))) - descent_vals
                coeff_vals[0] = initial_coeff - descent_vals[0]
                prev_jtheta = cur_jtheta
                loss = np.dot(X, coeff_vals) - train_data_label
                cur_jtheta = (np.sum(loss ** 2) + (lambd * sum([x**2 for x in coeff_vals])))/ (2 * m)
                print(f"Difference between consecutive costs: {abs(prev_jtheta - cur_jtheta)}\t", end="\r", flush=True)
            test_error = cost_function(test_data['Feature'], test_data[' Label'], len(test_data.index), coeff_vals) + ((lambd * sum([x**2 for x in coeff_vals])) / (2 * len(test_data.index)))
            curves.append(coeff_vals)
            # storing squared train and test error for plotting
            train_errors.append(cur_jtheta - ((lambd * sum([x**2 for x in coeff_vals])) / (2 * m)))
            test_errors.append(test_error - ((lambd * sum([x**2 for x in coeff_vals])) / (2 * len(test_data.index))))
            print(f"New Parameters: {coeff_vals}\t\t")
            print(f"Ridge Error on Test Data: {test_error}")
            print(f"Squared Error on Test Data: {test_error - ((lambd * sum([x**2 for x in coeff_vals])) / (2 * len(test_data.index)))}\n")
        result_curves.append(curves)
        ridge_training_errors.append(train_errors)
        ridge_test_errors.append(test_errors)
    
    # creating the graphs for all lambda for each of the two curves
    for i in range(2):
        # plotting test and train error for varying lambda
        labels = [0.25, 0.5, 0.75, 1]

        x = np.arange(len(labels))  # the label locations
        width = 0.5  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, ridge_training_errors[i], width, label='Ridge Train Error')
        rects2 = ax.bar(x + width/2, ridge_test_errors[i], width, label='Ridge Test Error')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Error')
        ax.set_xlabel('lambda -->')
        if i == 0:
            title = 'Best Curve: Train and Test Errors For Varying Lambda With Ridge'
        if i == 1:
            title = 'Worst Curve: Train and Test Errors For Varying Lambda With Ridge'
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        # display height of reactange
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{0:.7f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        fig.tight_layout()
        plt.show()

# Comment parts which are not to be run.
# Note: Parts 2a, 2b, 3a, 3b depend on part 1b and hence if they are run without part 1b, they'll load data from JSON files.
_1a_plot()
_1b_fitting()
_2a_plots()
_2b_plots()
_3a_lasso()
_3b_ridge()