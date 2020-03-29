# Shivam Kumar Jha
# 17CS30033
# Assignment 3
# ML-2020
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
import pickle

def main():
    df = pd.read_csv('../data/data_after_1a.csv')
    # df.head()

    # Initialize
    tfidf_trans = TfidfTransformer(smooth_idf=True, use_idf=True, norm='l2')

    # convert data in right format
    dtm_arr = df.as_matrix(columns=df.columns[1:])

    # fit to tfidf
    tfidf_trans.fit(dtm_arr)

    # calculate tdidf matrix
    dtm_arr = df.as_matrix(columns=df.columns[1:])
    tf_idf_vector = tfidf_trans.transform(dtm_arr)

    # save tfidf as a pickle
    with open('../data/tdidf_vector.pkl', 'wb') as f_open:
        pickle.dump(tf_idf_vector, f_open)


if __name__=="__main__":
    main()