import tensorflow as tf
import numpy as np


class W2V:
    def __init__(self, embedding, n_vocab, n_embedding=200, sub_threshold=1e-5, n_sampled=100, w2v_window_size=6):
        self.sub_threshold = sub_threshold
        self.n_sampled = n_sampled
        self.window_size = w2v_window_size
        self.n_vocab = n_vocab
        self.n_embedding = n_embedding

        self.a = tf.Variable(tf.random_uniform([n_embedding, ], -1, 1))
        self.a_embedding = tf.multiply(embedding, self.a)
        self.X = tf.placeholder(tf.int32, [None], name='w2v_inputs')
        self.Y = tf.placeholder(tf.int32, [None, None], name='w2v_labels')
        self.embed = tf.nn.embedding_lookup(self.a_embedding, self.X)
        self.softmax_w = tf.Variable(tf.truncated_normal((self.n_vocab, self.n_embedding)))  # create softmax weight
        self.softmax_b = tf.Variable(tf.zeros(self.n_vocab), name="softmax_bias")  # create softmax biases
        self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.a_embedding), 1, keepdims=True))
        self.normalized_embedding = self.a_embedding / self.norm

        self.loss = tf.nn.sampled_softmax_loss(
            weights=self.softmax_w,
            biases=self.softmax_b,
            labels=self.Y,
            inputs=self.embed,
            num_sampled=self.n_sampled,
            num_classes=self.n_vocab)

        self.cost = tf.reduce_mean(self.loss)
        self.train = tf.train.AdamOptimizer().minimize(self.cost)

    def get_target(self, words, idx):
        #Get a list of words in a window around an index.
        r = np.random.randint(1, self.window_size + 1)
        start = idx - r if (idx - r) > 0 else 0
        stop = idx + r
        target_words = set(words[start:idx] + words[idx + 1:stop + 1])

        return list(target_words)




