import tensorflow as tf
import os
import urllib2
import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import normalize
from numpy import linalg as LA


class SimilarityBase:

    def get_k_nn_column(self, k):
        def f(vector):
            v = np.zeros(len(vector))
            indices = np.argsort(vector)
            v[indices[0:k]] = vector[indices][0:k]
            return v
        return f

    def get_k_nn_column_rev(self, k):
        def f(vector):
            v = np.zeros(len(vector))
            rev_indices = np.argsort(vector)[::-1]
            v[rev_indices[0:k]] = vector[rev_indices][0:k]
            return v
        return f

    def sigmoidal_normalize(self, v):
        v = (v - min(v)) / (max(v) - min(v))
        return v


class DenseGaussianSimilarity(SimilarityBase):

    def __init__(self, sigma, standardized, sigmoidal_normalized):
        self.standardized = standardized
        self.sigma = sigma
        self.sigmoidal_normalized = sigmoidal_normalized

    def get_matrix(self, data):
        if self.standardized:
            data = normalize(data, axis = 0)

        m = np.exp( -( squareform(pdist(data, 'euclidean'))  ** 2 ) / (self.sigma ** 2))

        if self.sigmoidal_normalized:
            m = np.apply_along_axis(self.sigmoidal_normalize, 1, m)

        np.fill_diagonal(m,0.);
        return m

    def __repr__(self):
        description = "";
        if self.standardized:
            description += "First standarization is performed on data matrix - each feature is normally distributed after that. "

        description += "Euclidean distance matrix is computed - then it is transformed to Gaussian similarity matrix via Gaussian kernel with sigma = " + str(self.sigma) + ". "

        if self.sigmoidal_normalized:
            description += "Each matrix column is normalized linearly to have minimum value = 0 and maximum value = 1 (if C is a column of A then after normalization C = (C - min(C)) / (max(C) - min(C))). "
        description += "Diagonal of final matrix is then filled with zeros. "
        return description

class SparseGaussianSimilarity(SimilarityBase):

    def __init__(self, sigma, standardized, k, sigmoidal_normalized):

        self.standardized = standardized
        self.k = k
        self.sigma = sigma
        self.standardized = standardized
        self.sigmoidal_normalized = sigmoidal_normalized

    def get_matrix(self, data):
        if self.standardized:
            data = normalize(data, axis = 0)

        m = np.exp( -( squareform(pdist(data, 'euclidean'))  ** 2 ) / (self.sigma ** 2))
        m = np.apply_along_axis(self.get_k_nn_column_rev(self.k), 1, m)

        if self.sigmoidal_normalized:
            m = np.apply_along_axis(self.sigmoidal_normalize, 1, m)

        np.fill_diagonal(m,0.);

        return m

    def get_name(self):
        return "SparseGaussianSimilarity"

    def __repr__(self):
        description = "";
        if self.standardized:
            description += "First standarization is performed on data matrix - each feature is normally distributed after that. "

        description += "Euclidean distance matrix is computed - then it is transformed to Gaussian similarity matrix via Gaussian kernel with sigma = " + str(self.sigma) + ". K-nn matrix is build on resulting matrix with parameter k = " + str(self.k) + ". "

        if self.sigmoidal_normalized:
            description += "Each matrix column is normalized linearly to have minimum value = 0 and maximum value = 1 (if C is a column of A then after normalization C = (C - min(C)) / (max(C) - min(C))). "
        description += "Diagonal of final matrix is then filled with zeros. "
        return description


class DenseCosineSimilarity(SimilarityBase):

    def __init__(self, standardized, sigmoidal_normalized):
        self.standardized = standardized
        self.sigmoidal_normalized = sigmoidal_normalized

    def get_matrix(self, data):
        if self.standardized:
            data = normalize(data, axis = 0)

        m = squareform(pdist(data, 'cosine'))

        if self.sigmoidal_normalized:
            m = np.apply_along_axis(self.sigmoidal_normalize, 1, m)

        np.fill_diagonal(m,0.)

        return m

    def __repr__(self):
        description = "";
        if self.standardized:
            description += "First standarization is performed on data matrix - each feature is normally distributed after that. "

        description += "Cosine similarity matrix A is computed. "

        if self.sigmoidal_normalized:
            description += "Each matrix column is normalized linearly to have minimum value = 0 and maximum value = 1 (if C is a column of A then after normalization C = (C - min(C)) / (max(C) - min(C))). "
        description += "Diagonal of final matrix is then filled with zeros. "
        return description

class NormalizedDenseCosineSimilarity(SimilarityBase):
    def __init__(self, standardized, sigmoidal_normalized):
        self.standardized = standardized
        self.sigmoidal_normalized = sigmoidal_normalized

    def get_matrix(self, data):
        if self.standardized:
            data = normalize(data, axis = 0)

        m = squareform(pdist(data, 'cosine'))
        m = np.dot(LA.matrix_power(np.diag(np.sum(m,0)),-1), m)
        if self.sigmoidal_normalized:
            m = np.apply_along_axis(self.sigmoidal_normalize, 1, m)

        np.fill_diagonal(m,0.)
        return m

    def __repr__(self):
        description = "";
        if self.standardized:
            description += "First standarization is performed on data matrix - each feature is normally distributed after that. "

        description += "Cosine similarity matrix A is computed. Matrix is then row normalized D^-1 A. "

        if self.sigmoidal_normalized:
            description += "Each matrix column is normalized linearly to have minimum value = 0 and maximum value = 1 (if C is a column of A then after normalization C = (C - min(C)) / (max(C) - min(C))). "

        description += "Diagonal of final matrix is then filled with zeros. "
        return description


class SimilarityBasedDatasetIterator():

    def __init__(self, data, labels, similarity):
        self.data = data
        self.labels = labels
        self.matrix = similarity.get_matrix(data)
        self.data_size = self.matrix.shape[0]
        self.current_index = 0;

    def next_batch(self, n):
        return (self.matrix.transpose(), self.labels)

    def whole_dataset(self):
        return (self.matrix.transpose(), self.labels)

class DatasetIterator:

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def next_batch(self, n):
        return (self.data, self.labels)

    def whole_dataset(self):
        return (self.data, self.labels)


def download_wine_data(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

    if not os.path.exists(directory + '/' + 'wine.data'):
        response = urllib2.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data')
        data = response.read()
        f = open(directory + '/' + 'wine.data', 'wb')
        f.write(data)
        f.close()

    data = np.genfromtxt(directory + '/' + 'wine.data', delimiter=',')
    labels = data[:,0]
    data = data[:,1:]
    return (data, labels)


def read_wine_data(directory, similarity):
    data, labels = download_wine_data(directory)
    return SimilarityBasedDatasetIterator(data, labels, similarity)
