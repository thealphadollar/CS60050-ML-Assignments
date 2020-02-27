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

NOTE: This testing has been done on the entire data without split into train-test.

## Task 2 3

Run both the regressors with cross validation and print the average accuracy, precision and recall for both train and test data.

---SK Learn Logistic Regression---
Average Train Accuracy: 0.8808630393996247
Average Train Precision: 0.606871932173137
Average Train Recall: 0.34565772669220945
Average Test Accuracy: 0.8749218261413384
Average Test Precision: 0.5639203829680021
Average Test Recall: 0.3177321156773212

---Self Written Logistic Regression---
Average Train Accuracy: 0.8836772983114446
Average Train Precision: 0.6410100576767244
Average Train Recall: 0.31246831246831247
Average Test Accuracy: 0.8730456535334584
Average Test Precision: 0.5741363211951447
Average Test Recall: 0.29037740700144793


## Task 3 1

Created a manual class of Decision Tree Classifier and implemented required methods to use with
sklearn cross_validator.

NOTE: There is no runnable code in this python file since only the class is implemented

## Task 3 2

Tested the running of DecisionTreeClassifier from sklearn and prints the various values calculated 
by sklearn.

NOTE: This testing has been done on the entire data without split into train-test.

## Task 3 3

Run both the classifiers with cross validation and print the average accuracy, macro precision and macro
 recall for both train and test data.

---SK Learn Decision Tree Classifier---
Average Train Accuracy: 0.8861788617886179
Average Train Precision: 0.7955813251096694
Average Train Recall: 0.6166831618451797
Average Test Accuracy: 0.7961225766103815
Average Test Precision: 0.5356150186201415
Average Test Recall: 0.43883288486033406

---Self Written Decision Tree Classifier---                                      
Average Train Accuracy: 0.882426516572858
Average Train Precision: 0.7748472762337139
Average Train Recall: 0.6184477277802115
Average Test Accuracy: 0.7917448405253283
Average Test Precision: 0.4731587010076738
Average Test Recall: 0.409788914594477
