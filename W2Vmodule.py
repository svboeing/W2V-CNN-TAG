import tensorflow as tf
import numpy as np

class W2V():
    def __init__(self, n_vocab, n_embedding, sub_threshold, n_sampled, w2v_window_size):
        self.n_embedding = n_embedding
        self.n_vocab = n_vocab

        self.own_embedding = tf.Variable(tf.random_uniform((self.n_vocab, self.n_embedding), -1, 1, seed=123))
        self.sub_threshold = sub_threshold
        self.n_sampled = n_sampled
        self.window_size = w2v_window_size


        self.X = tf.placeholder(tf.int32, [None], name='w2v_inputs')
        self.Y = tf.placeholder(tf.int32, [None, None], name='w2v_labels')
        self.test_Y = tf.placeholder(tf.int32, [None], name='w2v_test_inputs')
        #self.embed = self.get_embed()


        self.softmax_w = tf.Variable(tf.truncated_normal((self.n_vocab, self.n_embedding), seed = 123))  # create softmax weight
        self.softmax_b = tf.Variable(tf.zeros(self.n_vocab), name="softmax_bias")  # create softmax biases
        self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.own_embedding), 1, keepdims=True)) #self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.a_embedding), 1, keepdims=True))
        self.normalized_embedding = self.own_embedding / self.norm #self.normalized_embedding = self.a_embedding / self.norm

        self.loss = tf.nn.sampled_softmax_loss(
            weights=self.softmax_w,
            biases=self.softmax_b,
            labels=self.Y,
            inputs=tf.nn.embedding_lookup(self.own_embedding, self.X),
            num_sampled=self.n_sampled,
            num_classes=self.n_vocab)



        self.cost = tf.reduce_mean(self.loss)
        self.optimizer = tf.train.AdamOptimizer() #learning_rate=1e-4, beta1=0.5, beta2=0.9
        self.train = self.optimizer.minimize(self.cost, var_list=(self.softmax_b, self.softmax_w, self.own_embedding))



        self.transposed_sotfmax_w = tf.transpose(self.softmax_w)
        self.logits = tf.nn.bias_add(tf.matmul(self.own_embedding, self.transposed_sotfmax_w), self.softmax_b)
        self.labels_one_hot = tf.one_hot(self.test_Y, self.n_vocab) #here Y must be flat
        self.test_loss = tf.nn.softmax_cross_entropy_with_logits_v2( #added
            labels=self.labels_one_hot,
            logits=tf.nn.embedding_lookup(self.logits, self.X) #there WAS NO LOOKUP AT ALL, DUMBASS
        )


    def get_target(self, words, idx):
        #Get a list of words in a window around an index.
        r = np.random.randint(1, self.window_size + 1)
        start = idx - r if (idx - r) > 0 else 0
        stop = idx + r
        target_words = set(words[start:idx] + words[idx + 1:stop + 1])

        return list(target_words)

    def dummy(self, input):
        return input


