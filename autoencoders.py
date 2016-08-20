import numpy as np
import tensorflow as tf


class Autoencoder:

    def __init__(self, encoder, decoder, optimizer, error_function, reg_error_component = None):

        self.encoder = encoder
        self.decoder = decoder
        self.error_function = error_function
        self.reg_error_component = reg_error_component
        self.error_function_node = self.error_function(self.decoder.get_network_output(),self.encoder.input_data)
        self.train_step = optimizer.minimize(self.error_function_node)


    def get_codes_nodes(self):
        return self.encoder.get_network_output()

    def get_codes_values(self, data):
        return self.sess.run(self.get_codes_nodes(), feed_dict={self.encoder.input_data: data });

    def get_reconstruction(self):
        return self.decoder.get_network_output()

    def init_tf_session(self):
        init = tf.initialize_all_variables();
        self.sess = tf.Session();
        self.sess.run(init);

    def run(self,dataset, nr_of_batches, batch_size):

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
