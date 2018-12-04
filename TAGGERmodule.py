import tensorflow as tf
import numpy as np

class TAGGER():
    def __init__(self, word_id, n_vocab, pref_id, tag_id, tagger_batch_size, n_embedding,
                 tagger_learning_rate, n_hidden_1,n_hidden_2, num_classes, n_tag_emb, n_suf_emb):

        self.n_embedding = n_embedding
        self.n_vocab = n_vocab
        self.own_embedding = tf.Variable(tf.random_uniform((self.n_vocab, self.n_embedding), -1, 1, seed=123))

        self.learning_rate = tagger_learning_rate

        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        #self.num_input = num_input #TODO -what is that?
        self.num_classes = num_classes
        self.n_tag_emb = n_tag_emb
        self.n_suf_emb = n_suf_emb
        self.xavier_init = tf.contrib.layers.xavier_initializer()



        self.weights = {
            'h1': tf.Variable(self.xavier_init([int(7 * self.n_embedding + 3 * self.n_tag_emb + 14 * self.n_suf_emb), self.n_hidden_1])),
            'h2': tf.Variable(self.xavier_init([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(self.xavier_init([self.n_hidden_2, self.num_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([self.n_hidden_1])),
            'b2': tf.Variable(tf.zeros([self.n_hidden_2])),
            'out': tf.Variable(tf.zeros([self.num_classes]))
        }

        self.suf = tf.Variable(tf.random_normal(shape=[len(pref_id) + 1, self.n_suf_emb], dtype="float32"), name='suffixes')
        self.pref = tf.Variable(tf.random_normal(shape=[len(pref_id) + 1, self.n_suf_emb], dtype="float32"), name='prefixes')
        self.tags = tf.Variable(tf.random_normal(shape=[len(tag_id) + 1, self.n_tag_emb], dtype="float32"), name='tags')

        #self.attention = False
        #self.lookup_matrix = embedding #if not self.attention else self.attented_embs
        self.X = tf.placeholder(tf.int32, [None, 24])  # 24 comes from supervector
        self.Y = tf.placeholder(tf.int32, [None, self.num_classes])  # num_classes is really len(tag_id)

        self.logits = self.neural_net(tagger_batch_size)

        self.test_logits = self.neural_net(1)
        self.test_output_labels = tf.argmax(self.test_logits, 1)

        self.correct_pred = tf.equal(self.test_output_labels, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


        # Define loss and optimizer
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.Y)) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.9)
        #self.no_attention_optimizer = tf.train.AdamOptimizer(learning_rate=self.l            beta2=0.9earning_rate)
        self.train = self.optimizer.minimize(self.loss, var_list=(
            self.weights['h1'], self.weights['h2'],self.weights['out'],self.biases['b1'], self.biases['b2'],
            self.biases['out'], self.suf, self.pref, self.tags, self.own_embedding))

        #self.output_labels = tf.argmax(tagger_logits, 1)

    def get_suf(self, some_word):
        return some_word[-2:]

    def get_pref(self, some_word):
        return some_word[:2]

    def supervector_maker(self, words_list, tags_list, pos, word_id, pref_id):
        if pos in range(3, len(words_list) - 3):
            supervector = np.array([word_id[words_list[pos - 3]], word_id[words_list[pos - 2]],
                                    word_id[words_list[pos - 1]], word_id[words_list[pos]],
                                    word_id[words_list[pos + 1]],
                                    word_id[words_list[pos + 2]], word_id[words_list[pos + 3]], tags_list[pos - 3],
                                    tags_list[pos - 2], tags_list[pos - 1], pref_id[self.get_pref(words_list[pos - 3])],
                                    pref_id[self.get_pref(words_list[pos - 2])], pref_id[self.get_pref(words_list[pos - 1])],
                                    pref_id[self.get_pref(words_list[pos])], pref_id[self.get_pref(words_list[pos + 1])],
                                    pref_id[self.get_pref(words_list[pos + 2])], pref_id[self.get_pref(words_list[pos + 3])],
                                    pref_id[self.get_suf(words_list[pos - 3])], pref_id[self.get_suf(words_list[pos - 2])],
                                    pref_id[self.get_suf(words_list[pos - 1])], pref_id[self.get_suf(words_list[pos])],
                                    pref_id[self.get_suf(words_list[pos + 1])], pref_id[self.get_suf(words_list[pos + 2])],
                                    pref_id[self.get_suf(words_list[pos + 3])]])

        if pos == 0:
            supervector = np.array([0, 0, 0, word_id[words_list[pos]],
                                    word_id[words_list[pos + 1]], word_id[words_list[pos + 2]],
                                    word_id[words_list[pos + 3]], 0, 0, 0, 0, 0, 0,
                                    pref_id[self.get_pref(words_list[pos])], pref_id[self.get_pref(words_list[pos + 1])],
                                    pref_id[self.get_pref(words_list[pos + 2])], pref_id[self.get_pref(words_list[pos + 3])], 0,
                                    0, 0,
                                    pref_id[self.get_suf(words_list[pos])], pref_id[self.get_suf(words_list[pos + 1])],
                                    pref_id[self.get_suf(words_list[pos + 2])], pref_id[self.get_suf(words_list[pos + 3])]])

        if pos == 1:
            supervector = np.array([0, 0, word_id[words_list[pos - 1]], word_id[words_list[pos]],
                                    word_id[words_list[pos + 1]], word_id[words_list[pos + 2]],
                                    word_id[words_list[pos + 3]], 0, 0, tags_list[pos - 1], 0, 0,
                                    pref_id[self.get_pref(words_list[pos - 1])], pref_id[self.get_pref(words_list[pos])],
                                    pref_id[self.get_pref(words_list[pos + 1])], pref_id[self.get_pref(words_list[pos + 2])],
                                    pref_id[self.get_pref(words_list[pos + 3])], 0, 0, pref_id[self.get_suf(words_list[pos - 1])],
                                    pref_id[self.get_suf(words_list[pos])], pref_id[self.get_suf(words_list[pos + 1])],
                                    pref_id[self.get_suf(words_list[pos + 2])], pref_id[self.get_suf(words_list[pos + 3])]])

        if pos == 2:
            supervector = np.array(
                [0, word_id[words_list[pos - 2]], word_id[words_list[pos - 1]], word_id[words_list[pos]],
                 word_id[words_list[pos + 1]], word_id[words_list[pos + 2]],
                 word_id[words_list[pos + 3]], 0, tags_list[pos - 2], tags_list[pos - 1],
                 0, pref_id[self.get_pref(words_list[pos - 2])], pref_id[self.get_pref(words_list[pos - 1])],
                 pref_id[self.get_pref(words_list[pos])], pref_id[self.get_pref(words_list[pos + 1])],
                 pref_id[self.get_pref(words_list[pos + 2])], pref_id[self.get_pref(words_list[pos + 3])],
                 0, pref_id[self.get_suf(words_list[pos - 2])], pref_id[self.get_suf(words_list[pos - 1])],
                 pref_id[self.get_suf(words_list[pos])], pref_id[self.get_suf(words_list[pos + 1])],
                 pref_id[self.get_suf(words_list[pos + 2])], pref_id[self.get_suf(words_list[pos + 3])]])

        if pos == len(words_list) - 1:
            supervector = np.array([word_id[words_list[pos - 3]], word_id[words_list[pos - 2]],
                                    word_id[words_list[pos - 1]], word_id[words_list[pos]], 0, 0, 0, tags_list[pos - 3],
                                    tags_list[pos - 2], tags_list[pos - 1], pref_id[self.get_pref(words_list[pos - 3])],
                                    pref_id[self.get_pref(words_list[pos - 2])], pref_id[self.get_pref(words_list[pos - 1])],
                                    pref_id[self.get_pref(words_list[pos])], 0, 0, 0, pref_id[self.get_suf(words_list[pos - 3])],
                                    pref_id[self.get_suf(words_list[pos - 2])], pref_id[self.get_suf(words_list[pos - 1])],
                                    pref_id[self.get_suf(words_list[pos])], 0, 0, 0])

        if pos == len(words_list) - 2:
            supervector = np.array([word_id[words_list[pos - 3]], word_id[words_list[pos - 2]],
                                    word_id[words_list[pos - 1]], word_id[words_list[pos]],
                                    word_id[words_list[pos + 1]], 0,
                                    0, tags_list[pos - 3], tags_list[pos - 2], tags_list[pos - 1],
                                    pref_id[self.get_pref(words_list[pos - 3])], pref_id[self.get_pref(words_list[pos - 2])],
                                    pref_id[self.get_pref(words_list[pos - 1])], pref_id[self.get_pref(words_list[pos])],
                                    pref_id[self.get_pref(words_list[pos + 1])], 0, 0, pref_id[self.get_suf(words_list[pos - 3])],
                                    pref_id[self.get_suf(words_list[pos - 2])], pref_id[self.get_suf(words_list[pos - 1])],
                                    pref_id[self.get_suf(words_list[pos])], pref_id[self.get_suf(words_list[pos + 1])], 0, 0])

        if pos == len(words_list) - 3:
            supervector = np.array([word_id[words_list[pos - 3]], word_id[words_list[pos - 2]],
                                    word_id[words_list[pos - 1]], word_id[words_list[pos]],
                                    word_id[words_list[pos + 1]],
                                    word_id[words_list[pos + 2]], 0, tags_list[pos - 3], tags_list[pos - 2],
                                    tags_list[pos - 1], pref_id[self.get_pref(words_list[pos - 3])],
                                    pref_id[self.get_pref(words_list[pos - 2])], pref_id[self.get_pref(words_list[pos - 1])],
                                    pref_id[self.get_pref(words_list[pos])], pref_id[self.get_pref(words_list[pos + 1])],
                                    pref_id[self.get_pref(words_list[pos + 2])], 0, pref_id[self.get_suf(words_list[pos - 3])],
                                    pref_id[self.get_suf(words_list[pos - 2])], pref_id[self.get_suf(words_list[pos - 1])],
                                    pref_id[self.get_suf(words_list[pos])], pref_id[self.get_suf(words_list[pos + 1])],
                                    pref_id[self.get_suf(words_list[pos + 2])], 0])

        return supervector

    def form_minibatch(self, words_list, tags_list, pos, batch_size, word_id, pref_id):
        matrix = []
        for counter in range(pos, pos + batch_size):
            line = self.supervector_maker(words_list, tags_list, counter,  word_id, pref_id)
            matrix.append(line)
        matrix = np.asarray(matrix).astype('int32')
        return matrix


    def form_feed(self, minibatch, batch_size, emb):  # now passing emb - instead of emb there goes concat of common and private
        #encoders,
        words_id_block = minibatch[:, :7]
        tags_id_block = minibatch[:, 7:10]
        pref_id_block = minibatch[:, 10:17]
        suf_id_block = minibatch[:, 17:24]



        words_lookup = tf.reshape(tf.nn.embedding_lookup(emb, words_id_block), (batch_size, -1)) # CHANGED




        tag_lookup = tf.reshape(tf.nn.embedding_lookup(self.tags, tags_id_block), (batch_size, -1))
        suf_lookup = tf.reshape(tf.nn.embedding_lookup(self.suf, suf_id_block), (batch_size, -1))
        pref_lookup = tf.reshape(tf.nn.embedding_lookup(self.pref, pref_id_block), (batch_size, -1))
        feed = tf.concat([words_lookup, tag_lookup, pref_lookup, suf_lookup], 1)
        return feed

    def form_encoder_feed(self, minibatch, batch_size, looked_up):  # now passing looked_up - there goes concat of common and private
        #encoders, with feed = TAGGER_net.X[:, :7]
        #words_id_block = minibatch[:, :7]
        tags_id_block = minibatch[:, 7:10]
        pref_id_block = minibatch[:, 10:17]
        suf_id_block = minibatch[:, 17:24]



        words_lookup = tf.reshape(looked_up, (batch_size, -1)) # CHANGED




        tag_lookup = tf.reshape(tf.nn.embedding_lookup(self.tags, tags_id_block), (batch_size, -1))
        suf_lookup = tf.reshape(tf.nn.embedding_lookup(self.suf, suf_id_block), (batch_size, -1))
        pref_lookup = tf.reshape(tf.nn.embedding_lookup(self.pref, pref_id_block), (batch_size, -1))
        feed = tf.concat([words_lookup, tag_lookup, pref_lookup, suf_lookup], 1)
        return feed

    def form_onehot_batch(self, labels, i, batch_size):
        matrix = []
        for counter in range(i, i + batch_size):
            hollow = np.zeros(17)
            hollow[labels[counter] - 1] = 1
            matrix.append(hollow)
        matrix = np.asarray(matrix).astype('int32')
        return matrix

    def neural_net(self, batch_size): # now passing emb NOOOOOOOOOOOOOOOOOOOOOOOOOOOOWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
        # Hidden fully connected layer with 256 neurons
        x_feed = self.form_feed(self.X, batch_size, self.own_embedding)
        layer_1 = tf.add(tf.matmul(x_feed, self.weights['h1']), self.biases['b1'])
        # Hidden fully connected layer with 64 neurons
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        return out_layer

    def encoder_neural_net(self, batch_size, looked_up): # now passing emb NOOOOOOOOOOOOOOOOOOOOOOOOOOOOWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
        # Hidden fully connected layer with 256 neurons
        x_feed = self.form_encoder_feed(self.X, batch_size, looked_up)
        layer_1 = tf.add(tf.matmul(x_feed, self.weights['h1']), self.biases['b1'])
        # Hidden fully connected layer with 64 neurons
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        return out_layer
