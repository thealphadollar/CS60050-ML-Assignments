# Shivam Kumar Jha
# 17CS30033
# Assignment 3
# ML-2020
import pickle as pkl
import numpy as np
from numpy.linalg import norm

MIN_DIST = 1e10
VEC_IND = 0
IND_IND = 1

class AgglomerativeClustering():
    """
    Agglomerative Clustering
    Recursively merges the pair of clusters that minimally increases
    a given linkage distance.
    """
    def __init__(self, n_clusters=8, affinity='cosine', linkage='single', path="../clusters/agglomerative.txt"):
        self._n_clusters = n_clusters
        self._affinity = affinity
        self._linkage = linkage
        self._path = path
        self._dist_mat = None
        
    @staticmethod
    def cos_sim(X,Y):
        """
        Return cosine similarity between two vectors.
        """
        return (X @ Y.T)/(norm(X)*norm(Y))
    
    def distance_matrix(self, X):
        """
        Computer matrix of exponential cosine distance between each data point.
        """
        self._dist_mat = np.zeros((X.shape[0], X.shape[0]))
#       calculating cosine distances
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if i!=j:
                    self._dist_mat[i][j] = np.exp(-1*self.cos_sim(X[i],X[j]))
        np.fill_diagonal(self._dist_mat, MIN_DIST)
#         print(self._dist_mat[545][545])
#         print(self._dist_mat.shape)
    
    def merge_clusters(self, X,Y):
        """
        Merge two clusters and return merged cluster.
        """
#         print(X[IND_IND],Y[IND_IND])
        merged_cluster = [[], []]
        merged_cluster[VEC_IND].extend(X[VEC_IND])
        merged_cluster[VEC_IND].extend(Y[VEC_IND])
        merged_cluster[IND_IND].extend(X[IND_IND])
        merged_cluster[IND_IND].extend(Y[IND_IND])
#         print("merged:",merged_cluster[IND_IND])
        return merged_cluster
    
    def single_clus_distance(self, X,Y):
        """
        Return single linkage minimum distance.
        
        Single linkage uses the minimum of the distances between all observations
        of the two sets.
        """
        x_fin, y_fin = None, None
        min_dist = MIN_DIST
#         print(len(X[IND_IND]), len(Y[IND_IND]))
        for i in range(0, len(X[IND_IND])):
#             print(X[IND_IND][i])
            for j in range(0, len(Y[IND_IND])):
#                 print(Y[IND_IND][j])
                if min_dist >= self._dist_mat[X[IND_IND][i]][Y[IND_IND][j]]:
#                     print("dist_mat",X[IND_IND][i],Y[IND_IND][j],self._dist_mat[X[IND_IND][i]][Y[IND_IND][j]])
                    min_dist = self._dist_mat[X[IND_IND][i]][Y[IND_IND][j]]
                    x_fin, y_fin = i,j
#         print("fin",x_fin,y_fin, min_dist)
        return min_dist
    
    def fit(self, X_arr):
        """
        Method to call to fit data.
        """
        init_clusters = [[[X_arr[i]], [i]] for i in range(X_arr.shape[0])]
        self.distance_matrix(X_arr)
        total_cur_clusters = X_arr.shape[0]
        while True:
            print(f"Number of clusters: {total_cur_clusters}", end='\r')
            clus_to_merge = None
            total_cur_clusters -= 1
            min_dist = MIN_DIST
            for i in range(total_cur_clusters):
                for j in range(i+1, total_cur_clusters):
                    cur_dist = self.single_clus_distance(init_clusters[i], init_clusters[j])
                    if min_dist >= cur_dist:
                        min_dist = cur_dist
                        clus_to_merge = [i,j]
#             print("min dist: ", min_dist, clus_to_merge[0], clus_to_merge[1])
            init_clusters.append(self.merge_clusters(init_clusters[clus_to_merge[0]], init_clusters[clus_to_merge[1]]))
            for index in sorted(clus_to_merge, reverse=True):
                del init_clusters[index]
            if total_cur_clusters==8:
                break
        self._results = init_clusters
#         print(init_clusters)
        self.save()
    
    def save(self):
        """
        Save the results in a sorted manner to ../clusters/agglomerative.txt
        """
        sorted_results = sorted(self._results, key= lambda x: min(x[IND_IND]))
        sorted_results = [sorted(x[IND_IND]) for x in sorted_results]
#         print(sorted_results)
        with open(self._path, 'w') as f_open:
            for result in sorted_results:
                f_open.write(','.join([str(x) for x in result]))
                f_open.write('\n')

def main():
    with open('../data/tdidf_vector.pkl', 'rb') as f_open:
        tfidf_matrix = pkl.load(f_open)
    # tfidf_matrix.toarray().shape
    agglo = AgglomerativeClustering()
    agglo.fit(tfidf_matrix.toarray())
    
if __name__=="__main__":
    main()