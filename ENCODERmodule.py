import tensorflow as tf
import numpy as np

def si_msi(ref, rec):
    return tf.losses.mean_squared_error(ref,rec) - (tf.reduce_sum(tf.subtract(ref,rec))/tf.size(ref))**2



class WDISC():
    def __init__(self, encoded_data_1, encoded_data_2, n_reduced, n_hidden, batch_size, LAMBDA): #instances of encoders are passed
        self.n_reduced = n_reduced
        self.n_hidden = n_hidden
        self.LAMBDA = LAMBDA
        self.xavier_init = tf.contrib.layers.xavier_initializer()
        self.weights = {
            'h1': tf.Variable(self.xavier_init([self.n_reduced, self.n_hidden])),
            'h2': tf.Variable(self.xavier_init([self.n_hidden, self.n_hidden])),
            'h3': tf.Variable(self.xavier_init([self.n_hidden, self.n_hidden])),
            'h4': tf.Variable(self.xavier_init([self.n_hidden, 1]))
        }

        self.biases = {
            'b1': tf.Variable(tf.zeros([self.n_hidden])),
            'b2': tf.Variable(tf.zeros([self.n_hidden])),
            'b3': tf.Variable(tf.zeros([self.n_hidden])),
            'b4': tf.Variable(tf.zeros([1]))
        }

        self.encoded_data_1 = encoded_data_1  # batch x n_reduced
        self.encoded_data_2 = encoded_data_2 # batch x n_reduced



        def neural_net(self, input):
            output = tf.nn.leaky_relu(tf.add(tf.matmul(input, self.weights['h1']), self.biases['b1']))
            output = tf.nn.leaky_relu(tf.add(tf.matmul(output, self.weights['h2']), self.biases['b2']))
            output = tf.nn.leaky_relu(tf.add(tf.matmul(output, self.weights['h3']), self.biases['b3']))
            output = tf.add(tf.matmul(output, self.weights['h4']), self.biases['b4'])
            return tf.reshape(output, [-1])  #TODO - is this meant to be local function?

        self.wdisc_1 = neural_net(self, self.encoded_data_1)
        self.wdisc_2 = neural_net(self, self.encoded_data_2)

        self.alpha = tf.random_uniform(
            shape=[batch_size, 1], #batch_size tf.shape(self.encoded_data_1)[0]
            minval=0.,
            maxval=1.
        ) # - will it be recounted on each step? TODO - check - YES

        self.interpolates = self.alpha * self.encoded_data_1 + ((1 - self.alpha) * self.encoded_data_2)
        self.wdisc_interpolates = neural_net(self, self.interpolates)
        self.gradients = tf.gradients(self.wdisc_interpolates, [self.interpolates])[0] # TODO - make sure it does what we want - seems like yes
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1]))
        self.gradient_penalty = tf.reduce_mean((self.slopes - 1) ** 2)

        self.disc_cost = tf.reduce_mean(self.wdisc_1) - tf.reduce_mean(self.wdisc_2) + \
                         self.LAMBDA * self.gradient_penalty

        self.train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.5,
            beta2=0.9
        ).minimize(
            self.disc_cost,
            var_list=(self.weights['h1'], self.weights['h2'],self.weights['h3'],self.weights['h4'],self.biases['b1'],
                      self.biases['b2'], self.biases['b3'], self.biases['b4'])
        )
        # sgd momentum optimizer






class ENCODER():
    def __init__(self, n_embedding, n_hidden = 256, n_reduced = 128):

        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.n_reduced = n_reduced

        self.xavier_init = tf.contrib.layers.xavier_initializer()
        self.weights = {
            'h1': tf.Variable(self.xavier_init(
                [self.n_embedding, self.n_hidden])),
            'h2': tf.Variable(self.xavier_init([self.n_hidden, self.n_hidden])),
            'h3': tf.Variable(self.xavier_init([self.n_hidden, self.n_hidden])),
            'h4': tf.Variable(self.xavier_init([self.n_hidden, self.n_reduced]))
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([self.n_hidden])),
            'b2': tf.Variable(tf.zeros([self.n_hidden])),
            'b3': tf.Variable(tf.zeros([self.n_hidden])),
            'b4': tf.Variable(tf.zeros([self.n_reduced]))
        }

    def neural_net(self, input, embedding):

        layer = tf.add(tf.matmul(tf.nn.embedding_lookup(embedding, input), self.weights['h1']), self.biases['b1'])
            # Hidden fully connected layer
        layer = tf.add(tf.matmul(layer, self.weights['h2']), self.biases['b2'])
        layer = tf.add(tf.matmul(layer, self.weights['h3']), self.biases['b3'])
        layer = tf.add(tf.matmul(layer, self.weights['h4']), self.biases['b4'])
        return layer

class COMMON_ENCODER(ENCODER):
    def __init__(self, embedding_1, embedding_2, n_embedding, n_hidden = 256, n_reduced = 128):#embeddings are to be passed
        ENCODER.__init__(self, n_embedding, n_hidden, n_reduced)
        self.embedding_1 = embedding_1
        self.embedding_2 = embedding_2
        self.X = tf.placeholder(tf.int32, [None, ])



        self.encoded_data_1 = ENCODER.neural_net(self, self.X, self.embedding_1) #common encoders share weights, like discriminators
        self.encoded_data_2 = ENCODER.neural_net(self, self.X, self.embedding_2)

class PRIVATE_ENCODER(ENCODER):
    def __init__(self, embedding, n_embedding, n_hidden=256,
                 n_reduced=128):  # embeddings are to be passed
        ENCODER.__init__(self, n_embedding, n_hidden, n_reduced)
        self.embedding = embedding
        self.X = tf.placeholder(tf.int32, [None, ])


        self.encoded_data = ENCODER.neural_net(self, self.X,
                                                 self.embedding) #private encoders do not share weights


class DECODER():
    def __init__(self, encoded_data_1, encoded_data_2, reference_1, reference_2, n_reduced, n_embedding, n_hidden):
        self.encoded_data_1 = encoded_data_1  # batch x 2n_reduced
        self.encoded_data_2 = encoded_data_2  # batch x 2n_reduced
        self.reference_1 = reference_1
        self.reference_2 = reference_2
        self.n_embedding = n_embedding
        self.n_reduced = n_reduced
        self.n_hidden = n_hidden
        self.xavier_init = tf.contrib.layers.xavier_initializer()
        self.weights = {
            'h1': tf.Variable(self.xavier_init(
                [2*self.n_reduced, self.n_hidden])),
            'h2': tf.Variable(self.xavier_init([self.n_hidden, self.n_hidden])),
            'h3': tf.Variable(self.xavier_init([self.n_hidden, self.n_hidden])),
            'h4': tf.Variable(self.xavier_init([self.n_hidden, self.n_embedding]))
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([self.n_hidden])),
            'b2': tf.Variable(tf.zeros([self.n_hidden])),
            'b3': tf.Variable(tf.zeros([self.n_hidden])),
            'b4': tf.Variable(tf.zeros([self.n_embedding]))
        }

        def neural_net(self, input):
            layer = tf.add(tf.matmul(input, self.weights['h1']), self.biases['b1'])
            # Hidden fully connected layer
            layer = tf.add(tf.matmul(layer, self.weights['h2']), self.biases['b2'])
            layer = tf.add(tf.matmul(layer, self.weights['h3']), self.biases['b3'])
            layer = tf.add(tf.matmul(layer, self.weights['h4']), self.biases['b4'])
            return layer

        def si_msi(ref, rec):
            return tf.losses.mean_squared_error(ref, rec) - (tf.reduce_sum(tf.subtract(ref, rec)) / tf.to_float(tf.size(ref))) ** 2

        self.recon_1 = neural_net(self, self.encoded_data_1)
        self.recon_2 = neural_net(self, self.encoded_data_2)

        self.loss = si_msi(self.reference_1, self.recon_1) + si_msi(self.reference_2, self.recon_2)



