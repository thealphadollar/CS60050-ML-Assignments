from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from os import path
import pandas as pd

# ignore warnings of uncovergence because we are not setting convergence criteria
import warnings
warnings.filterwarnings('ignore')

from task_3_1 import DTClassifier

def main():
    src_dir = path.join(path.dirname(path.realpath(__file__)), '..', 'data')

    # load data
    train_data = pd.read_csv(path.join(src_dir, 'dataset_B.csv'), sep=';')
    train_data_label = train_data['quality'].to_numpy()
    train_data_feature = train_data.loc[:, train_data.columns != 'quality'].to_numpy()
    
    # load regressors
    sk_dtc = DecisionTreeClassifier(criterion='entropy', min_samples_split=10)
    self_dtc = DTClassifier()

    # set required scoring parameters
    scoring = {'accu': 'accuracy',
           'prec': 'precision_macro',
           'reca': 'recall_macro'}

    # cross validate sklearn with 3 fold and calculating train accuracy, precision and recall as well
    scores = cross_validate(sk_dtc, train_data_feature, train_data_label, cv=3, scoring=scoring, return_train_score=True)
    print("---SK Learn Decision Tree Classifier---")
    print(f"Average Train Accuracy: {scores['train_accu'].mean()}")
    print(f"Average Train Precision: {scores['train_prec'].mean()}")
    print(f"Average Train Recall: {scores['train_reca'].mean()}")
    print(f"Average Test Accuracy: {scores['test_accu'].mean()}")
    print(f"Average Test Precision: {scores['test_prec'].mean()}")
    print(f"Average Test Recall: {scores['test_reca'].mean()}")

    # cross validate manual logistic regressor with 3 fold and calculating train accuracy, precision and recall as well
    scores = cross_validate(self_dtc, train_data_feature, train_data_label, cv=3, scoring=scoring, return_train_score=True)
    print("---Self Written Decision Tree Classifier---                                      ")
    print(f"Average Train Accuracy: {scores['train_accu'].mean()}")
    print(f"Average Train Precision: {scores['train_prec'].mean()}")
    print(f"Average Train Recall: {scores['train_reca'].mean()}")
    print(f"Average Test Accuracy: {scores['test_accu'].mean()}")
    print(f"Average Test Precision: {scores['test_prec'].mean()}")
    print(f"Average Test Recall: {scores['test_reca'].mean()}")
    

if __name__=='__main__':
    main()