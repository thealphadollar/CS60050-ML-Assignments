# Shivam Kumar Jha
# 17CS30033
# Assignment 3
# ML-2020
from sklearn.decomposition import PCA
import pickle as pkl
from task_b import AgglomerativeClustering
from task_c import KMeans

def main():
    with open('../data/tdidf_vector.pkl', 'rb') as f_open:
        tfidf_matrix = pkl.load(f_open)
    pca = PCA(n_components=100)
    tfidf_array = pca.fit_transform(tfidf_matrix.toarray())
    # tfidf_array.shape

    # Agglomerative Reduced
    agglo = AgglomerativeClustering(path="../clusters/agglomerative_reduced.txt")
    agglo.fit(tfidf_matrix.toarray())

    # KMeans Reduced
    kmeans = KMeans(path="../clusters/kmeans_reduced.txt")
    kmeans.fit(tfidf_matrix.toarray())

if __name__=="__main__":
    main()