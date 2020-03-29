# Shivam Kumar Jha
# 17CS30033
# Assignment 3
# ML-2020
import pickle as pkl
import numpy as np
from random import randrange
from numpy.linalg import norm

MIN_DIST = 1e10
class KMeans():
    """
    Agglomerative Clustering
    Recursively merges the pair of clusters that minimally increases
    a given linkage distance.
    """
    def __init__(self, n_clusters=8, iters=300, path="clusters/kmeans.txt"):
        self._n_clusters = n_clusters
        self._iters = iters
        self._path = path
        self._dist_mat = None
        
    def distance_matrix(self, X):
        """
        Computer matrix of exponential cosine distance between each data point.
        """
        self._dist_mat = np.zeros((X.shape[0], X.shape[0]))
#       calculating cosine distances
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if i!=j:
                    self._dist_mat[i][j] = self.cos_sim_dist(X[i],X[j])
        np.fill_diagonal(self._dist_mat, MIN_DIST)
#         print(self._dist_mat[545][545])
#         print(self._dist_mat.shape)
        
    @staticmethod
    def cos_sim_dist(X,Y):
        """
        Return exponential cosine similarity between two vectors.
        """
        return np.exp(-1 * ((X @ Y.T)/(norm(X)*norm(Y))))
    
    def init_centroids(self, X):
        """
        Create random centroids.
        """
        np.random.RandomState(randrange(0, 1e4))
        self._centroids = X[np.random.choice(X.shape[0], self._n_clusters, replace=False)]
        
    def closest_centroid(self, X):
        """
        Return index of centroid closest to the given X vector.
        """
        min_dist_list = [self.cos_sim_dist(X, Y) for Y in self._centroids]
        return min_dist_list.index(min(min_dist_list))   
            
    
    def fit(self, X_arr):
        """
        Method to call to fit data.
        """
        self.init_centroids(X_arr)
        self.distance_matrix(X_arr)
        doc_labels = np.zeros((X_arr.shape[0], 1))
        for i in range(self._iters):
            centroid_points = [[] for _ in range(self._n_clusters)]
            print(f"Iteration number {i}", end="\r")
            for index, arr in enumerate(X_arr):
#                 assign closest centroid
                doc_labels[index] = self.closest_centroid(arr)
                for num in doc_labels[index]:
                    centroid_points[int(num)].append(index)
                
    #             compute new centroids
            new_centroids = np.zeros((self._n_clusters, X_arr.shape[1]))
            for ind in range(self._n_clusters):
                new_centroids[ind, :] = np.mean(np.take(X_arr, centroid_points[ind], axis=0), axis=0)
            
#             break if no change detected
            if np.all(self._centroids == new_centroids):
                print("breaking early...")
                break
            self._centroids = new_centroids
        self._results = [[] for _ in range(self._n_clusters)]
        for i, ele in enumerate(doc_labels):
            for num in ele:
                # print(num)
                self._results[int(num)].append(i)
        # print(self._results)
        self.save()
    
    def save(self):
        """
        Save the results in a sorted manner to clusters/agglomerative.txt
        """
        sorted_results = sorted(self._results, key= lambda x: min(x))
        sorted_results = [sorted(x) for x in sorted_results]
#         print(sorted_results)
        with open(self._path, 'w') as f_open:
            for result in sorted_results:
                f_open.write(','.join([str(x) for x in result]))
                f_open.write('\n')

def main():
    with open('data/tdidf_vector.pkl', 'rb') as f_open:
        tfidf_matrix = pkl.load(f_open)
    # tfidf_matrix.toarray().shape
    kmeans = KMeans(iters=1000)
    kmeans.fit(tfidf_matrix.toarray())

if __name__=="__main__":
    main()