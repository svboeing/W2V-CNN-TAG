import tensorflow as tf
import numpy as np
from ATTENTIONmodule import ATTENTION

class W2V(ATTENTION):
    def __init__(self, embedding, adv_loss, n_vocab, n_embedding, attention_size, sub_threshold, n_sampled, w2v_window_size): #attention_size is added
        ATTENTION.__init__(self, embedding, n_embedding, n_vocab, attention_size) # TODO - add training step
        self.sub_threshold = sub_threshold
        self.n_sampled = n_sampled
        self.window_size = w2v_window_size
        self.adv_loss = adv_loss

        #self.n_embedding = n_embedding - IN ATTENTION

        #self.a = tf.Variable(tf.random_uniform([n_embedding, ], -1, 1))
        #self.a_embedding = tf.multiply(embedding, self.a)

        #self.a_embedding = tf.nn.softmax(tf.nn.tanh(embedding), axis=0) * embedding
        #self.attention = False
        #self.lookup_matrix = embedding #if not self.attention else self.attented_embs
        self.X = tf.placeholder(tf.int32, [None], name='w2v_inputs')
        self.Y = tf.placeholder(tf.int32, [None, None], name='w2v_labels')
        self.test_Y =tf.placeholder(tf.int32, [None], name='w2v_test_inputs')
        #self.embed = self.get_embed()


        self.softmax_w = tf.Variable(tf.truncated_normal((self.n_vocab, self.n_embedding)))  # create softmax weight
        self.softmax_b = tf.Variable(tf.zeros(self.n_vocab), name="softmax_bias")  # create softmax biases
        self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding), 1, keepdims=True)) #self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.a_embedding), 1, keepdims=True))
        self.normalized_embedding = self.embedding / self.norm #self.normalized_embedding = self.a_embedding / self.norm

        self.attention_loss = tf.nn.sampled_softmax_loss(
            weights=self.softmax_w,
            biases=self.softmax_b,
            labels=self.Y,
            inputs=tf.nn.embedding_lookup(self.attented_embs, self.X),
            num_sampled=self.n_sampled,
            num_classes=self.n_vocab)

        self.own_attention_loss = tf.nn.sampled_softmax_loss(
            weights=self.softmax_w,
            biases=self.softmax_b,
            labels=self.Y,
            inputs=tf.nn.embedding_lookup(self.own_attented_embs, self.X),
            num_sampled=self.n_sampled,
            num_classes=self.n_vocab)

        self.attention_cost = tf.reduce_mean(self.attention_loss)
        self.own_attention_cost = tf.reduce_mean(self.own_attention_loss)
        self.attention_optimizer = tf.train.AdamOptimizer()
        #self.own_attention_optimizer = tf.train.AdamOptimizer()
        self.attention_train = self.attention_optimizer.minimize(self.attention_cost, var_list=(self.softmax_b, self.softmax_w, self.W, self.b, self.u, self.embedding))
        self.own_attention_train = self.attention_optimizer.minimize(self.own_attention_cost, var_list=(self.softmax_b, self.softmax_w, self.W, self.b, self.u, self.own_embedding))
        self.adv_train = self.attention_optimizer.minimize(tf.negative(self.adv_loss), var_list=(embedding))

        self.transposed_sotfmax_w = tf.transpose(self.softmax_w)
        self.logits = tf.nn.bias_add(tf.matmul(self.attented_embs, self.transposed_sotfmax_w), self.softmax_b) # TEST IS ATTENTED
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




