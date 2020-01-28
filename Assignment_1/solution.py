# Shivam Kumar Jha
# 17CS30033
# The functions are written in the order of the questions and solution to, for example 1a is named as _1a_plot

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from copy import copy

# loading training data
try:
    train_data = pd.read_csv('train.csv')
except FileNotFoundError:
    train_data = pd.read_csv(input('Enter path to training data: '))
    
# loading testing data
try:
    test_data = pd.read_csv('test.csv')
except FileNotFoundError:
    test_data = pd.read_csv(input('Enter path to test data: '))    

# global variable to store training error, test error and predicted parameters
# index 0 corresponds to polynomial of degree 1 and so on
all_train_error = []
all_test_error = []
predicted_parameters = []
# highest degree polynomial to build starting from 0
till_n = 9
learning_rate = 0.05

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
    # calculating sums
    sum = 0.0
    for ind in range(m):
        pol_val = 0.0
        for index, coeff in enumerate(coeffs):
            pol_val += coeff * pow(X.iloc[ind], index)
        sum += pow(pol_val-Y.iloc[ind], 2)
    return sum / (2*m)

# calculates the descent
def calc_descent(X, Y, m, coeffs, x_index, learning_rate):
    # calculating sums
    sum = 0.0
    for ind in range(m):
        pol_val = 0.0
        for index, coeff in enumerate(coeffs):
            pol_val += coeff * pow(X.iloc[ind], index)
        sum += (pol_val-Y.iloc[ind])*(X.iloc[ind] ** x_index)
    return learning_rate * (sum/m)

# returns best, worst curve based on training data error
def get_extreme_curves():
    best_train_curve = None
    worst_train_curve = None
    min_train_error = min(all_train_error)
    max_train_error = max(all_train_error)
    for index, train_error in enumerate(all_train_error):
        if train_error == min_train_error:
            best_train_curve = predicted_parameters[index]
        if train_error == max_train_error:
            worst_train_curve = predicted_parameters[index]
    return best_train_curve, worst_train_curve

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
        # setting initial value of the parameters
        print(f"=========== Polynomial Degree {n} ===========")
        coeff_vals = [0 for _ in range(n+1)]
        prev_jtheta = None
        cur_jtheta = cost_function(train_data['Feature'], train_data[' Label'], len(train_data.index), coeff_vals)
        # setting convergence when the difference between consecutive error is less than .001
        while (prev_jtheta is None or abs(prev_jtheta - cur_jtheta) > 0.001):
            descent_vals = []
            for x_power in range(n+1):
                descent_vals.append(calc_descent(train_data['Feature'], train_data[' Label'], len(train_data.index), coeff_vals, x_power, learning_rate))
            for index, descent_val in enumerate(descent_vals):
                coeff_vals[index] = coeff_vals[index] - descent_val
            prev_jtheta = cur_jtheta
            cur_jtheta = cost_function(train_data['Feature'], train_data[' Label'], len(train_data.index), coeff_vals)
        predicted_parameters.append(coeff_vals)
        all_train_error.append(cur_jtheta)
        test_error = cost_function(test_data['Feature'], test_data[' Label'], len(test_data.index), coeff_vals)
        all_test_error.append(test_error)
        print(f"Parameters: {coeff_vals}")
        print(f"Squared Error on Test Data: {test_error}\n")

# plotting the predicted polynomials, answer to 2a
def _2a_plots():
    for index, coeffs in enumerate(predicted_parameters):
        plt.scatter(train_data['Feature'], poly_calc(train_data['Feature'], coeffs))
        plt.xlabel("Feature")
        plt.ylabel("Label")
        plt.title(f"Polynomial of Degree {index+1}")
        plt.show()

# plotting the squared error for training and test data for the values of n
def _2b_plots():
    labels = [str(x) for x in range(1,till_n+1)]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

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
            ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()
    
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
        curves = []
        train_errors = []
        test_errors = []
        for lambd in [.25, .5, .75, 1]:
            print(f"=========== Lasso Regularisation {curve_type}: Polynomial Degree {n} Lambda {lambd} ===========")
            coeff_vals = copy(curve)
            prev_jtheta = None
            cur_jtheta = cost_function(train_data['Feature'], train_data[' Label'], len(train_data.index), coeff_vals) + (lambd * sum([abs(x) for x in coeff_vals]))
            # setting convergence when the difference between consecutive error is less than .001
            while (prev_jtheta is None or abs(prev_jtheta - cur_jtheta) > 0.001):
                descent_vals = []
                for x_power in range(n+1):
                    descent_vals.append(calc_descent(train_data['Feature'], train_data[' Label'], len(train_data.index), coeff_vals, x_power, learning_rate))
                for index, descent_val in enumerate(descent_vals):
                    coeff_vals[index] = coeff_vals[index] - descent_val
                prev_jtheta = cur_jtheta
                cur_jtheta = cost_function(train_data['Feature'], train_data[' Label'], len(train_data.index), coeff_vals) + (lambd * sum([abs(x) for x in coeff_vals]))
            test_error = cost_function(test_data['Feature'], test_data[' Label'], len(test_data.index), coeff_vals) + (lambd * sum([abs(x) for x in coeff_vals]))
            curves.append(coeff_vals)
            train_errors.append(cur_jtheta)
            test_errors.append(test_error)
            print(f"Parameters: {coeff_vals}")
            print(f"Lasso Error on Test Data: {test_error}\n")
            print(f"Squared Error on Test Data: {test_error - (lambd * sum([abs(x) for x in coeff_vals]))}\n")
        result_curves.append(curves)
        lasso_training_errors.append(train_errors)
        lasso_test_errors.append(test_errors)
    
    for i in range(2):
        # plotting test and train error for varying lambda
        labels = [0.25, 0.5, 0.75, 1]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

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
                ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        fig.tight_layout()
        plt.show()


def _3b_ridge():
    best_curve, worst_curve = get_extreme_curves()
    result_curves = []
    ridge_training_errors = []
    ridge_test_errors = []
    is_best = True
    for curve in [best_curve, worst_curve]:
        n = len(curve) - 1
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
            coeff_vals = copy(curve)
            prev_jtheta = None
            cur_jtheta = cost_function(train_data['Feature'], train_data[' Label'], len(train_data.index), coeff_vals) + (lambd * sum([x**2 for x in coeff_vals]))
            # setting convergence when the difference between consecutive error is less than .001
            while (prev_jtheta is None or abs(prev_jtheta - cur_jtheta) > 0.001):
                descent_vals = []
                for x_power in range(n+1):
                    descent_vals.append(calc_descent(train_data['Feature'], train_data[' Label'], len(train_data.index), coeff_vals, x_power, learning_rate))
                for index, descent_val in enumerate(descent_vals):
                    coeff_vals[index] = coeff_vals[index] - descent_val
                prev_jtheta = cur_jtheta
                cur_jtheta = cost_function(train_data['Feature'], train_data[' Label'], len(train_data.index), coeff_vals) + (lambd * sum([x**2 for x in coeff_vals]))
            test_error = cost_function(test_data['Feature'], test_data[' Label'], len(test_data.index), coeff_vals) + (lambd * sum([x**2 for x in coeff_vals]))
            curves.append(coeff_vals)
            train_errors.append(cur_jtheta)
            test_errors.append(test_error)
            print(f"Parameters: {coeff_vals}")
            print(f"Ridge Error on Test Data: {test_error}\n")
            print(f"Squared Error on Test Data: {test_error - (lambd * sum([x**2 for x in coeff_vals]))}\n")
        result_curves.append(curves)
        ridge_training_errors.append(train_errors)
        ridge_test_errors.append(test_errors)
    
    for i in range(2):
        # plotting test and train error for varying lambda
        labels = [0.25, 0.5, 0.75, 1]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

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
                ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        fig.tight_layout()
        plt.show()


_1a_plot()
_1b_fitting()
_2a_plots()
_2b_plots()
_3a_lasso()
_3b_ridge()