import tensorflow as tf
from numpy import linalg as LA
import numpy as np
from sklearn.preprocessing import normalize
from scipy.cluster.vq import kmeans, vq

def least_squares_reconstruction_error(x, recontructions):
    return tf.reduce_mean(tf.square(x - recontructions))

def sparsity_regularization_error_component(beta, rho, codes):
    return tf.mul(beta ,  tf.reduce_sum(kl_divergance(rho, tf.reduce_mean(codes,0))))

def l2_regularization_error_component(lambda_param, weights):
    result =  tf.constant(0.)
    for weight in weights:
        result = result + tf.nn.l2_loss(weight)
    return tf.mul(lambda_param , result)


def spectral_clustering(W):
    D_ = LA.matrix_power(np.diag(np.sum(W, 0)),-1/2)
    L = np.dot(np.dot(D_, W), D_)
    eigenvectors = np.linalg.eig(L)[1]
    X = np.real(eigenvectors[:,0:3])
    X = normalize(X, axis=1)
    kmeans_centroids,_ =  kmeans(X, 3, 400, 40)
    kmeans_idx, _ = vq(X, kmeans_centroids)
    return kmeans_idx
