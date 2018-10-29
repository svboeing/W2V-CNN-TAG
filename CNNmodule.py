import tensorflow as tf
from ATTENTIONmodule import ATTENTION

class CNN(ATTENTION):
    def __init__(self, embedding, adv_loss, n_embedding, n_vocab, attention_size, sent_max_features, sent_maxlen, n_filters,
                 sent_kernel_size, sent_hidden_dims, sent_learning_rate, sent_width, sent_eval_batch_size): #attention_size is added
        ATTENTION.__init__(self,embedding, n_embedding, n_vocab, attention_size)
        # params
        self.adv_loss = adv_loss
        self.max_features = sent_max_features
        self.maxlen = sent_maxlen
        self.n_filters = n_filters
        self.kernel_size = sent_kernel_size
        self.hidden_dims = sent_hidden_dims
        self.learning_rate = sent_learning_rate
        self.width = sent_width
        #self.n_embedding = n_embedding in ATTENTION
        # graph nodes
        #self.a = tf.Variable(tf.random_uniform([self.n_embedding, ], -1, 1))
        #self.a_embedding = tf.multiply(embedding, self.a)

        self.X = tf.placeholder("int32", [None, self.maxlen])
        self.Y = tf.placeholder("float32", [None, ])

        #self.attention = False
        #self.lookup_matrix = embedding #if not self.attention else self.attented_embs

        self.xavier_init = tf.contrib.layers.xavier_initializer()
        #                    word_embs = tf.Variable(xavier_init([max_features, embedding_dims]))
        self.filters = tf.Variable(self.xavier_init([self.width, self.n_embedding, self.n_filters]))  # embedding_dims
        self.attention_logits = tf.squeeze(self.neural_net(self.attented_embs), [1])
        self.own_attention_logits = tf.squeeze(self.neural_net(self.own_attented_embs), [1])
        self.batch_prediction = tf.nn.sigmoid(self.attention_logits)                      #prediction is kinda attented
        self.eval_batch_size = sent_eval_batch_size

        # Define loss and optimizer
        self.attention_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.attention_logits, labels=self.Y)) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.own_attention_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.own_attention_logits, labels=self.Y)) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.attention_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #self.no_attention_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.attention_train = self.attention_optimizer.minimize(self.attention_loss, var_list=(self.filters, self.W, self.b, self.u, self.embedding))
        self.own_attention_train = self.attention_optimizer.minimize(self.own_attention_loss, var_list=(self.filters, self.W, self.b, self.u, self.own_embedding))
        self.adv_train = self.attention_optimizer.minimize(tf.negative(self.adv_loss, name = 'piss'), var_list=(self.embedding), name = 'fuck_me') #self.W, self.b, self.u embedding

    def neural_net(self, emb):                                     # added emb
        x = tf.nn.embedding_lookup(emb, self.X) #x = tf.nn.embedding_lookup(self.attented_embs, self.X)
        x = tf.layers.dropout(inputs=x, rate=0.2)
        x = tf.nn.conv1d(value=x, filters=self.filters, stride=1, padding='VALID')
        x = tf.nn.relu(features=x)
        x = tf.layers.max_pooling1d(inputs=x, pool_size=self.maxlen - 2, strides=1,
                                    padding='VALID')
        x = tf.squeeze(x, [1])
        x = tf.layers.dense(inputs=x, units=self.hidden_dims, activation='relu')
        x = tf.layers.dropout(inputs=x, rate=0.2)
        x = tf.layers.dense(inputs=x, units=1)
        return x

