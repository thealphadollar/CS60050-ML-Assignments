# Shivam Kumar Jha
# 17CS30033
# Assignment 3
# ML-2020
import pickle as pkl
import pandas as pd
import numpy as np
from csv import reader

ENUM = {
    'Buddhism': 0,
    'TaoTeChing': 1,
    'Upanishad': 2,
    'YogaSutra': 3,
    'BookOfProverb': 4,
    'BookOfEcclesiastes': 5,
    'BookOfEccleasiasticus': 6,
    'BookOfWisdom': 7
}

class NMI():
    
    def __init__(self):
        self._load_orig()
        self._calc_enp_class()
    
    def _load_orig(self):
        """
        Load original label data.
        """
        df = pd.read_csv('data/data_after_1a.csv')
        self._labels = df.iloc[:, 0]
#         print(self._labels)
        for i,val in enumerate(self._labels):
            self._labels[i] = ENUM[val]
#         print(self._labels)
    
    def calculate(self, path):
        """
        Calculate NMI for the given file.
        """
        self._load_data_from_file(path)
        cluster_entropy = self._enp_clst()
        cond_entropy = self._enp_cond()
        # print(self._enp_class, cond_entropy, cluster_entropy)
        nmi = 2*(self._enp_class - cond_entropy)/(self._enp_class+cluster_entropy)
#         print(f"NMI: {nmi}")
        return nmi
        
    def _load_data_from_file(self, path):
        """
        Load data from file containing data as list of lists.
        """
        with open(path, 'r') as f_open:
            csv_reader = reader(f_open)
#             for row in csv_reader:
#                 print(row)
            data_list = list(csv_reader)
        self._pred_list = data_list
        self._total_preds = 0 
        for pred in self._pred_list:
            self._total_preds += len(pred)
#         self.pred_labels = np.zeros(shape=self._labels.shape)
    
    def _calc_enp_class(self):
        """
        Calculate entropy for the initial labelled data.
        """
        class_list = [self._labels[self._labels == i] for i in range(8)]
        self._enp_class = -1 * sum([((len(x)/self._labels.shape[0])*np.log2(len(x)/self._labels.shape[0])) 
                                    for x in class_list])
#         print(self._enp_class)
        
    def _enp_clst(self):
        """
        Calculate entropy of the given cluster.
        """
        class_list = self._pred_list
        return -1 * sum([((len(x)/self._total_preds)*np.log2(len(x)/self._total_preds)) 
                                    for x in class_list])

    def _enp_cond(self):
        """
        Calculate conditional entropy.
        """
        return sum([self._enp_cond_helper(x, self._total_preds) 
                                    for x in self._pred_list])
        
    def _enp_cond_helper(self, cur_clust, num_ele):
        unq_cnt_dict = {
            0:0,
            1:0,
            2:0,
            3:0,
            4:0,
            5:0,
            6:0,
            7:0
        }
        for ind,val in enumerate(cur_clust):
            unq_cnt_dict[self._labels[int(val)]] += 1
        for k,v in list(unq_cnt_dict.items()):
            if v == 0:
                del unq_cnt_dict[k]
        tempVal = -1 * sum([(unq_cnt_dict[key]/len(cur_clust))*np.log2(unq_cnt_dict[key]/len(cur_clust)) 
                           for key in unq_cnt_dict.keys()])
        return (len(cur_clust)/num_ele)*tempVal

def main():
    with open('data/tdidf_vector.pkl', 'rb') as f_open:
        tfidf_matrix = pkl.load(f_open)
    nmi = NMI()
    print(f'NMI Score For Agglomerative Clustering: {nmi.calculate("clusters/agglomerative.txt")}')
    print(f'NMI Score For Reduced Agglomerative Clustering: {nmi.calculate("clusters/agglomerative_reduced.txt")}')
    print(f'NMI Score For KMeans Clustering: {nmi.calculate("clusters/kmeans.txt")}')
    print(f'NMI Score For Reduced KMeans Clustering: {nmi.calculate("clusters/kmeans_reduced.txt")}')

if __name__=="__main__":
    main()