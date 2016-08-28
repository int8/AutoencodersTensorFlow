import numpy as np
from tf_utils import *
from io_utils import *
from tqdm import tqdm
import tensorflow as tf
from autoencoders import *
from scipy.cluster.vq import kmeans, vq
from sklearn.metrics import normalized_mutual_info_score
from tensorflow.examples.tutorials.mnist import input_data



def run_wine_experiment_no_sparsity(similarity, encoder_network_graph, decoder_network_graph, nr_of_experiments,  nr_of_batches):
    return run_wine_experiment_core(similarity, encoder_network_graph, decoder_network_graph, nr_of_experiments,  nr_of_batches)

def run_wine_experiment_sparsity(similarity, encoder_network_graph, decoder_network_graph, nr_of_experiments,  nr_of_batches, rho, beta):
    return run_wine_experiment_core(similarity, encoder_network_graph, decoder_network_graph, nr_of_experiments,  nr_of_batches, 'KL', rho, beta)

def run_wine_experiment_sparsity_hoyer(similarity, encoder_network_graph, decoder_network_graph, nr_of_experiments,  nr_of_batches):
    return run_wine_experiment_core(similarity, encoder_network_graph, decoder_network_graph, nr_of_experiments,  nr_of_batches, 'Hoyer')


def run_wine_experiment_greedy_core(similarity, nr_of_experiments,  nr_of_batches, sparse = None, beta = None, rho = None):

    wine = read_wine_data("WINE_data/", similarity)
    X = wine.next_batch(1)[0]
    indx =  spectral_clustering(X)
    spectral_nmi  =  (normalized_mutual_info_score(wine.next_batch(1)[1], indx))
    print spectral_nmi
    results = []

    for i in tqdm(range(nr_of_experiments)):

        g = GreedyAutoencoder(tf.train.AdamOptimizer())
        if sparse == 'KL':
            g.add_sparse_layer(178, 64, tf.nn.sigmoid, tf.nn.sigmoid, beta, rho)
            g.add_sparse_layer(64, 32, tf.nn.sigmoid, tf.nn.sigmoid, beta, rho)
            g.add_sparse_layer(32, 10, tf.nn.sigmoid, tf.nn.sigmoid, beta, rho)
        elif sparse == 'Hoyer':
            g.add_sparse_layer_hoyer(178, 64, tf.nn.sigmoid, tf.nn.sigmoid)
            g.add_sparse_layer_hoyer(64, 32, tf.nn.sigmoid, tf.nn.sigmoid)
            g.add_sparse_layer_hoyer(32, 10, tf.nn.sigmoid, tf.nn.sigmoid)
        else:
            g.add_layer(178, 64, tf.nn.sigmoid, tf.nn.sigmoid)
            g.add_layer(64, 32, tf.nn.sigmoid, tf.nn.sigmoid)
            g.add_layer(32, 10, tf.nn.sigmoid, tf.nn.sigmoid)

        g.build_error_function()
        encoded_dataset  = g.train(wine, nr_of_batches, 178)

        centroids,_ =  kmeans(encoded_dataset,  3, 400)

        idx,_ = vq(encoded_dataset,centroids)
        results.append((normalized_mutual_info_score(wine.next_batch(178)[1], idx)))

        print normalized_mutual_info_score(wine.next_batch(178)[1], idx)

    return {"desc": str(similarity), "spectral_nmi": spectral_nmi,  "experiment_results": results}


def run_wine_experiment_core(similarity, encoder_network_graph, decoder_network_graph, nr_of_experiments,  nr_of_batches, sparse = None,  rho = None, beta = None):

    wine = read_wine_data("WINE_data/", similarity)
    X = wine.next_batch(1)[0]
    indx =  spectral_clustering(X)
    spectral_nmi  =  (normalized_mutual_info_score(wine.next_batch(1)[1], indx))
    results = []
    print spectral_nmi
    
    for i in tqdm(range(nr_of_experiments)):

        if sparse == 'KL':
             autoencoder = SparseAutoencoder(encoder_network_graph, decoder_network_graph, least_squares_reconstruction_error,  beta, rho)
        elif sparse == 'Hoyer':
             autoencoder = SparseAutoencoderHoyer(encoder_network_graph, decoder_network_graph,least_squares_reconstruction_error)
        else:
             autoencoder = Autoencoder(encoder_network_graph, decoder_network_graph, least_squares_reconstruction_error)

        autoencoder.build_error_function(tf.train.AdamOptimizer())
        autoencoder.init_tf_session()
        autoencoder.train(wine, nr_of_batches = nr_of_batches, batch_size = 178)
        encoded_dataset =  autoencoder.get_codes_values(wine.next_batch(178)[0])

        sparsity = autoencoder.get_sparsity_value(wine)
        autoencoder.close_tf_session()
        centroids,_ =  kmeans(encoded_dataset,  3, 400)

        idx,_ = vq(encoded_dataset,centroids)
        results.append((normalized_mutual_info_score(wine.next_batch(178)[1], idx)))


        print normalized_mutual_info_score(wine.next_batch(178)[1], idx), sparsity

    return {"desc": str(similarity), "spectral_nmi": spectral_nmi,  "experiment_results": results}
