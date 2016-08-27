import numpy as np
import tensorflow as tf
from tf_utils import *
from io_utils import DatasetIterator

class GreedyAutoencoder:

    def __init__(self, optimizer):
        self.optimizer = optimizer;
        self.autoencoders = []

    def add_layer(self, input_size, size, activation_encoder, activation_decoder):
        encoder_network_graph, decoder_network_graph = self.build_autocoder_components(input_size, size, activation_encoder, activation_decoder)
        autoencoder = Autoencoder(encoder_network_graph, decoder_network_graph, self.optimizer,  least_squares_reconstruction_error)
        self.autoencoders.append(autoencoder)


    def add_sparse_layer(self, input_size, size, activation_encoder, activation_decoder, beta, rho ):
        encoder_network_graph, decoder_network_graph = self.build_autocoder_components(input_size, size, activation_encoder, activation_decoder)
        autoencoder = SparseAutoencoder(encoder_network_graph, decoder_network_graph, self.optimizer,  least_squares_reconstruction_error,  beta, rho)
        self.autoencoders.append(autoencoder)


    def build_autocoder_components(self, input_size, size, activation_encoder, activation_decoder):
        encoder_network_graph = MultiLayerPerceptron(input_size, [], size)
        encoder_network_graph.build_computational_graph([activation_encoder])

        decoder_network_graph = MultiLayerPerceptron(size, [],  input_size, encoder_network_graph.get_network_output())
        decoder_network_graph.build_computational_graph([activation_decoder])

        return (encoder_network_graph, decoder_network_graph)

    def train(self, dataset, nr_of_batches, batch_size):

        current_dataset = dataset
        for i in xrange(len(self.autoencoders)):
            self.autoencoders[i].init_tf_session()
            self.autoencoders[i].train(current_dataset, nr_of_batches, batch_size)
            current_dataset = self.autoencoders[i].get_codes_values_iterator(current_dataset)
            self.autoencoders[i].close_tf_session()

        return current_dataset.whole_dataset()[0]

class Autoencoder:

    def __init__(self, encoder, decoder, optimizer, error_function):

        self.encoder = encoder
        self.decoder = decoder
        self.error_function = error_function
        self.error_function_node = self.error_function(self.decoder.get_network_output(),self.encoder.input_data)
        self.train_step = optimizer.minimize(self.error_function_node)



    def get_codes_nodes(self):
        return self.encoder.get_network_output()

    def get_codes_values(self, data):
        return self.sess.run(self.get_codes_nodes(), feed_dict={self.encoder.input_data: data });

    def get_codes_values_iterator(self, data):

        X = self.sess.run(self.get_codes_nodes(), feed_dict={self.encoder.input_data: data.whole_dataset()[0] });
        return DatasetIterator(X, [])


    def get_reconstruction(self):
        return self.decoder.get_network_output()

    def init_tf_session(self):
        init = tf.initialize_all_variables();
        self.sess = tf.Session();
        self.sess.run(init);

    def train(self,dataset, nr_of_batches, batch_size):

        for i in xrange(1,nr_of_batches):
            current_batch = dataset.next_batch(batch_size)
            self.sess.run(self.train_step, feed_dict={self.encoder.input_data: current_batch[0] });

    def close_tf_session(self):
        self.sess.close()


class MultiLayerPerceptron:


    def __init__(self, input_size, hidden_layer_sizes, output_size, input_data = None):

        self.input_data = tf.placeholder(tf.float32, [None, input_size]) if input_data == None else input_data
        self.weights = [];
        self.biases = [];
        self.graph_built = False

        all_sizes_array = np.hstack([input_size, hidden_layer_sizes, output_size]) if len(hidden_layer_sizes) > 0 else np.array([input_size, output_size]);

        for i in range(len(all_sizes_array) - 1):
            W = tf.Variable(tf.truncated_normal([all_sizes_array[i], all_sizes_array[i + 1]]))
            b = tf.Variable(tf.zeros(all_sizes_array[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

    def build_computational_graph(self, activations):

        if  len(activations) != len(self.weights):
            raise Exception('number of activation functions should match network size')

        self.network_flow = []
        current_layer_output = self.input_data
        for i in range(len(self.weights)):
            current_layer_output = activations[i](tf.matmul(current_layer_output, self.weights[i]) + self.biases[i])
            self.network_flow.append(current_layer_output)

        self.graph_built = True

    def get_network_output(self):

        if not self.graph_built:
            raise Exception("Graph has not been built - please build computational graph using build_computational_graph function")

        return self.network_flow[-1]



class SparseAutoencoder(Autoencoder):

    def __init__(self, encoder, decoder, optimizer, error_function, beta, rho):
         Autoencoder.__init__(self, encoder, decoder, optimizer, error_function)
         self.error_function_node = self.error_function_node + self.sparsity_regularization_error_component(beta, rho, encoder.get_network_output())

    def sparsity_regularization_error_component(self, beta, rho, codes):
         return tf.mul(beta ,  tf.reduce_sum(self.kl_divergance(rho, tf.reduce_mean(codes,0))))

    def log_func(self, a, b):
        return tf.mul(a, tf.log(tf.div(a ,b)))

    def kl_divergance(self, rho, rho_hat):
        invrho = tf.sub(tf.constant(1.), rho)
        invrhohat = tf.sub(tf.constant(1.), rho_hat)
        logrho = tf.add(self.log_func(rho,rho_hat), self.log_func(invrho, invrhohat))
        return logrho
