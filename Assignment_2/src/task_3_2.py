import numpy as np
import pandas as pd
from os import path
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def main():
    src_dir = path.join(path.dirname(path.realpath(__file__)), '..', 'data')

    train_data = pd.read_csv(path.join(src_dir, 'dataset_B.csv'), sep=';')
    # print(train_data)
    train_data_label = train_data['quality'].to_numpy()
    train_data_feature = train_data.loc[:, train_data.columns != 'quality'].to_numpy()
    # print(train_data_feature.shape)

    # create DTC with info gain and min split 10
    dtc = DecisionTreeClassifier(criterion='entropy', min_samples_split=10)
    dtc.fit(train_data_feature, train_data_label)

    # make predictions on the training data
    predictions = dtc.predict(train_data_feature)

    print(f"Accuracy Train Data: {metrics.accuracy_score(train_data_label, predictions)}")
    print(f"Precision Train Data: {metrics.precision_score(train_data_label, predictions, average='macro')}")
    print(f"Recall Train Data: {metrics.recall_score(train_data_label, predictions, average='macro')}")
    

if __name__ == '__main__':
    main()