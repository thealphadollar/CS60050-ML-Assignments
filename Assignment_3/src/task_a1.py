# Shivam Kumar Jha
# 17CS30033
# Assignment 3
# ML-2020
import pandas as pd

def main():
    # input data
    df = pd.read_csv('data/AllBooks_baseline_DTM_Labelled.csv')
    df.rename({"Unnamed: 0": "Religion"}, axis='columns', inplace=True)
    # df.head()

    # removing _Chx
    df.Religion = df.Religion.apply(lambda x: x[:x.rindex('_')])
    # df.tail()
    # df.shape

    # remove row 13 (index 0)
    df.drop(df.index[[13,]], inplace=True)
    # df.shape

    # save new csv data
    df.to_csv('data/data_after_1a.csv', sep=',', index=False)

if __name__=="__main__":
    main()