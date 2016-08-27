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
    return run_wine_experiment_core(similarity, encoder_network_graph, decoder_network_graph, nr_of_experiments,  nr_of_batches, True, rho, beta)


def run_wine_experiment_core(similarity, encoder_network_graph, decoder_network_graph, nr_of_experiments,  nr_of_batches, sparse = None, rho = None, beta = None):

    wine = read_wine_data("WINE_data/", similarity)
    X = wine.next_batch(1)[0]
    indx =  spectral_clustering(X)
    spectral_nmi  =  (normalized_mutual_info_score(wine.next_batch(1)[1], indx))
    results = []

    for i in tqdm(range(nr_of_experiments)):
        if sparse:
             autoencoder = SparseAutoencoder(encoder_network_graph, decoder_network_graph, tf.train.AdamOptimizer(),  least_squares_reconstruction_error,  beta, rho)
        else:
             autoencoder = Autoencoder(encoder_network_graph, decoder_network_graph, tf.train.AdamOptimizer(),  least_squares_reconstruction_error)
        autoencoder.init_tf_session()
        autoencoder.run(wine, nr_of_batches = nr_of_batches, batch_size = 178)
        encoded_dataset =  autoencoder.get_codes_values(wine.next_batch(178)[0])
        autoencoder.close_tf_session()
        centroids,_ =  kmeans(encoded_dataset,  3, 400)

        idx,_ = vq(encoded_dataset,centroids)
        results.append((normalized_mutual_info_score(wine.next_batch(178)[1], idx)))

        print normalized_mutual_info_score(wine.next_batch(178)[1], idx)

    return {"desc": str(similarity), "spectral_nmi": spectral_nmi,  "experiment_results": results}
