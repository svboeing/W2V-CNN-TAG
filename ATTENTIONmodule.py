import tensorflow as tf

class ATTENTION:
    def __init__(self, embedding, n_embedding, n_vocab, attention_size):
        self.n_embedding = n_embedding
        self.n_vocab = n_vocab
        self.embedding = embedding
        self.own_embedding = tf.Variable(tf.random_uniform((self.n_vocab, self.n_embedding), -1, 1, seed = 123))
        self.attention_size = attention_size
        self.W = tf.Variable(tf.random_normal([self.n_embedding, self.attention_size], stddev=0.1)) #dtype = 'float64'
        self.b = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
        self.u = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
        self.layer_1 = tf.tanh(tf.matmul(self.embedding, self.W) + self.b)
        self.alphas = (n_vocab / 1000) * tf.nn.softmax(tf.tensordot(self.layer_1, self.u, axes = 1)) #reduce att_size
        self.attented_embs = (tf.multiply(self.embedding, self.alphas[:, None]))
        self.own_attented_embs = (tf.multiply(self.own_embedding, self.alphas[:, None]))
