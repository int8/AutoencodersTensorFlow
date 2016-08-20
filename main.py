import numpy as np
from tf_utils import *
from io_utils import *
import tensorflow as tf
from autoencoders import *
from experiments import *
import json
from pprint import pprint
import hashlib


encoder_network_graph = MultiLayerPerceptron(178, [64, 32], 10)
encoder_network_graph.build_computational_graph([tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid])

decoder_network_graph = MultiLayerPerceptron(10, [32],  178, encoder_network_graph.get_network_output())
decoder_network_graph.build_computational_graph([tf.nn.sigmoid, tf.nn.sigmoid])

similarities = [
    # DenseCosineSimilarity(standardized = True,  sigmoidal_normalized = True),
    # DenseCosineSimilarity(standardized = True,  sigmoidal_normalized = False),
    # NormalizedDenseCosineSimilarity(standardized = True,  sigmoidal_normalized = True),
    # NormalizedDenseCosineSimilarity(standardized = True,  sigmoidal_normalized = False),
    # DenseGaussianSimilarity(sigma = 0.9, standardized = True, sigmoidal_normalized = True),
    # DenseGaussianSimilarity(sigma = 0.5, standardized = True, sigmoidal_normalized = True),
    # DenseGaussianSimilarity(sigma = 0.1, standardized = True, sigmoidal_normalized = True),
    # DenseGaussianSimilarity(sigma = 0.9, standardized = True, sigmoidal_normalized = False),
    # DenseGaussianSimilarity(sigma = 0.5, standardized = True, sigmoidal_normalized = False),
    # DenseGaussianSimilarity(sigma = 0.1, standardized = True, sigmoidal_normalized = False),
    # SparseGaussianSimilarity(sigma = 0.1, standardized = True, k = 30, sigmoidal_normalized = True),
    # SparseGaussianSimilarity(sigma = 0.5, standardized = True, k = 30, sigmoidal_normalized = True),
    SparseGaussianSimilarity(sigma = 0.9, standardized = True, k = 30, sigmoidal_normalized = True),
    SparseGaussianSimilarity(sigma = 0.1, standardized = True, k = 30, sigmoidal_normalized = False),
    SparseGaussianSimilarity(sigma = 0.5, standardized = True, k = 30, sigmoidal_normalized = False),
    SparseGaussianSimilarity(sigma = 0.9, standardized = True, k = 30, sigmoidal_normalized = False)
]

all_results = []
for similarity in similarities:
    pprint(str(similarity))
    expertiment_results = run_wine_experiment(similarity, encoder_network_graph, decoder_network_graph, 50, 9000)
    all_results.append(expertiment_results)

    with open('results/chunks_64_32_10_32/' + (hashlib.md5(str(similarity))).hexdigest() + '.txt', 'a+') as outfile:
        json.dump(expertiment_results, outfile)

#
# with open('results/data.txt', 'a+') as outfile:
#     json.dump(all_results, outfile)
