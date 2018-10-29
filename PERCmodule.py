import tensorflow as tf

class PERC:
    def __init__(self, embedding, n_embedding, n_hidden, perc_learning_rate, n_sampled): # 1st arg added - embedddddding
        self.n_embedding = n_embedding
        self.embedding = embedding
        self.learning_rate = perc_learning_rate #percs percs percs get me outta control
        self.n_hidden = n_hidden
        self.num_classes = 3 #three nets are involved but still)))
        self.n_sampled = n_sampled
        self.xavier_init = tf.contrib.layers.xavier_initializer()
        self.weights = {
            'h1': tf.Variable(self.xavier_init([self.n_embedding, self.n_hidden])),
            'h2': tf.Variable(self.xavier_init([self.n_hidden, self.num_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([self.n_hidden])),
            'b2': tf.Variable(tf.zeros([self.num_classes]))
        }

        self.X = tf.placeholder(tf.int32, [None, ])#self.n_embedding])
        self.Y = tf.placeholder(tf.int32, [None, 1])
        self.one_hot_Y = tf.one_hot(self.Y, self.num_classes)

        self.layer_1 = tf.add(tf.matmul(tf.nn.embedding_lookup(self.embedding, self.X), self.weights['h1']), self.biases['b1'])
        self.layer_2 = tf.add(tf.matmul(self.layer_1, self.weights['h2']), self.biases['b2'])

        #self.loss = tf.nn.sampled_softmax_loss(
        #    weights=self.weights['h2'],
        #    biases=self.biases['b2'],
        #    labels=self.one_hot_Y,
        #    inputs=self.layer_1,
        #    num_sampled=self.n_sampled,
        #    num_classes=self.num_classes,
        #    name = 'classifier_loss')

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits = self.layer_2,
            labels = self.one_hot_Y
        ))


        self.train = tf.train.AdamOptimizer().minimize(self.cost, var_list=(self.weights['h1'], self.weights['h2'], self.biases['b1'], self.biases['b2']))

