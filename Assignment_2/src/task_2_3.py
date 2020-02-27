from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from os import path
import pandas as pd

# ignore warnings of uncovergence because we are not setting convergence criteria
import warnings
warnings.filterwarnings('ignore')

from task_2_1 import LogisticRegressor


def main():
    src_dir = path.join(path.dirname(path.realpath(__file__)), '..', 'data')

    # load data
    train_data = pd.read_csv(path.join(src_dir, 'dataset_A.csv'), sep=';')
    train_data_label = train_data['quality'].to_numpy()
    train_data_feature = train_data.loc[:, train_data.columns != 'quality'].to_numpy()
    
    # load regressors
    sk_logregressor = LogisticRegression(penalty='none', solver='saga')
    self_logregpressor = LogisticRegressor()

    # set required scoring parameters
    scoring = {'accu': 'accuracy',
           'prec': 'precision',
           'reca': 'recall'}

    # cross validate sklearn with 3 fold and calculating train accuracy, precision and recall as well
    scores = cross_validate(sk_logregressor, train_data_feature, train_data_label, cv=3, scoring=scoring, return_train_score=True)
    print("---SK Learn Logistic Regression---")
    print(f"Average Train Accuracy: {scores['train_accu'].mean()}")
    print(f"Average Train Precision: {scores['train_prec'].mean()}")
    print(f"Average Train Recall: {scores['train_reca'].mean()}")
    print(f"Average Test Accuracy: {scores['test_accu'].mean()}")
    print(f"Average Test Precision: {scores['test_prec'].mean()}")
    print(f"Average Test Recall: {scores['test_reca'].mean()}")

    # cross validate manual logistic regressor with 3 fold and calculating train accuracy, precision and recall as well
    scores = cross_validate(self_logregpressor, train_data_feature, train_data_label, cv=3, scoring=scoring, return_train_score=True)
    print("---Self Written Logistic Regression---                                      ")
    print(f"Average Train Accuracy: {scores['train_accu'].mean()}")
    print(f"Average Train Precision: {scores['train_prec'].mean()}")
    print(f"Average Train Recall: {scores['train_reca'].mean()}")
    print(f"Average Test Accuracy: {scores['test_accu'].mean()}")
    print(f"Average Test Precision: {scores['test_prec'].mean()}")
    print(f"Average Test Recall: {scores['test_reca'].mean()}")
    

if __name__=='__main__':
    main()