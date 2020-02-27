To run the program, please use the following steps.

- Install Python 3.6 with pip
- Run `pip3 install -r requirements.txt`
- Run `python3 src/task_{number}_{alphabet}`. For example, `python3 src/task_1_a.py` for 1.a task.

Once the solution is running, all the requested graphs (if any) as per the question will come up one by one, please close the graphs to move forward.

## Task 1 a

Generates `data/dataset_A.csv` as per the description given in the assignment.

## Task 1 b

Generates `data/dataset_B.csv` as per the description given in the assignment.

Apart from the last bucket, it has been assumed that the end of the range is 
not included in the current bucket but the start of the range is included.

## Task 2 1

Created a manual class of Logistic Regressor with Gradient Descent and implemented required methods to use with
sklearn cross_validator.

NOTE: There is no runnable code in this python file since only the class is implemented

## Task 2 2

Tested the running of LogisticRegressor from sklearn and prints the parameters as well as error calculated 
by method prescribed in class.

## Task 2 3

Run both the regressors with cross validation and print the average accuracy, precision and recall for both train and test data.

---SK Learn Logistic Regression---
Average Train Accuracy: 0.8808630393996247
Average Train Precision: 0.7553184763756969
Average Train Recall: 0.655281563773114
Average Test Accuracy: 0.8749218261413384
Average Test Precision: 0.7319885643724788
Average Test Recall: 0.6400490563610913

---Self Written Logistic Regression---
Average Train Accuracy: 0.8836772983114446
Average Train Precision: 0.7707684928748993
Average Train Recall: 0.6426513828478065
Average Test Accuracy: 0.8730456535334584
Average Test Precision: 0.7354169832248685
Average Test Recall: 0.627613894780464