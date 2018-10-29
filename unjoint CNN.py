import numpy as np
import tensorflow as tf
from collections import Counter
import random
from keras.datasets import imdb
from keras.preprocessing import sequence
from sklearn.metrics import f1_score

n_embedding = 200
num_global_batches = 5000
epochs = 15

# Set CNN parameters:
sent_max_features = 5000
sent_maxlen = 400
# sent_batch_size = 32
# sent_embedding_dims = 50
n_filters = 250
sent_kernel_size = 3
sent_hidden_dims = 250
sent_learning_rate = 0.003
sent_training_steps = 2
sent_width = 3


unique_sorted_words = np.load("/home/boeing/PycharmProjects/CNN/JOINT_sorted_words.npy")
unique = set(unique_sorted_words)

vocab_to_int = {}
int_to_vocab = {}

for i, word in enumerate(unique_sorted_words):
    vocab_to_int[word] = i+1 #!!!!!!!!!!!!!!!! # NOW 0 IS RESERVED ONCE AGAIN - SAME AS TAGGER_S
    int_to_vocab[i+1] = word #!!!!!!!!!!!!!!!!!!!


# CNN PART

#  SHHHHHHHHHHHHHHHHHHHHIiiiiiiIIIIIIIIIIIIIIIIIIIIIIIIIIITTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
(sent_x_train_raw, sent_y_train), (sent_x_test_raw, sent_y_test) = imdb.load_data(num_words=sent_max_features, index_from=3)
# print(len(x_train), 'train sequences')
# print(len(x_test), 'test sequences')



imdb_w_to_id = imdb.get_word_index()
imdb_w_to_id = {k:(v + 3) for k, v in imdb_w_to_id.items()}
imdb_w_to_id["<PAD>"] = 0
imdb_w_to_id["<START>"] = 1
imdb_w_to_id["<UNK>"] = 2

imdb_id_to_w = {value:key for key, value in imdb_w_to_id.items()}

sent_x_train, sent_x_test = [], []

for i in range(len(sent_x_train_raw)):
    sent_x_train.append([vocab_to_int[imdb_id_to_w[id]] for id in sent_x_train_raw[i] if imdb_id_to_w[id] in unique])
    sent_x_test.append([vocab_to_int[imdb_id_to_w[id]] for id in sent_x_test_raw[i] if imdb_id_to_w[id] in unique])

#now imdb dataset consists of correct ids of words that appear in text8 -lookup will work

# print('Pad sequences (samples x time)') SHHHHHHHHHHHHHHHHHHHHIiiiiiiIIIIIIIIIIIIIIIIIIIIIIIIIIITTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
sent_x_train = sequence.pad_sequences(sent_x_train, maxlen=sent_maxlen)
sent_x_test = sequence.pad_sequences(sent_x_test, maxlen=sent_maxlen)


def sent_neural_net(x):
    x = tf.nn.embedding_lookup(embedding, x)
    x = tf.layers.dropout(inputs=x, rate=0.2)
    x = tf.nn.conv1d(value=x, filters=filters, stride=1, padding='VALID')
    x = tf.nn.relu(features=x)
    x = tf.layers.max_pooling1d(inputs=x, pool_size=sent_maxlen - 2, strides=1,
                                padding='VALID')
    x = tf.squeeze(x, [1])
    x = tf.layers.dense(inputs=x, units=250, activation='relu')
    x = tf.layers.dropout(inputs=x, rate=0.2)
    x = tf.layers.dense(inputs=x, units=1)
    return x

# COMMON EMBEDDINGS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
train_graph = tf.Graph()
n_vocab = len(int_to_vocab) + 1

with train_graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1)) #cuz 0 is reserved!!!!!!!!!!



    # CNN graph nodes
    #sent_a = tf.Variable(tf.random_uniform([n_embedding,], -1, 1))
    #sent_a_embedding = tf.multiply(embedding, sent_a)
    sent_X = tf.placeholder("int32", [None, sent_maxlen])
    sent_Y = tf.placeholder("float32", [None, ])
    xavier_init = tf.contrib.layers.xavier_initializer()
    #                                      word_embs = tf.Variable(xavier_init([max_features, embedding_dims]))
    filters = tf.Variable(xavier_init([sent_width, n_embedding, n_filters]))  # embedding_dims


# COMMON


with train_graph.as_default():


    sent_logits = tf.squeeze(sent_neural_net(sent_X), [1])
    sent_batch_prediction = tf.nn.sigmoid(sent_logits)

    # Define loss and optimizer
    sent_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=sent_logits, labels=sent_Y))
    sent_optimizer = tf.train.AdamOptimizer(learning_rate=sent_learning_rate)
    sent_train_op = sent_optimizer.minimize(sent_loss_op)


def get_joint_batches(sent_x_train, sent_y_train):

    sent_batch_size = len(sent_x_train) // num_global_batches
    #tagger_batch_size = len(tagger_train_words) // num_global_batches
    # only full batches

    sent_x_train = sent_x_train[:num_global_batches * sent_batch_size]
    sent_y_train = sent_y_train[:num_global_batches * sent_batch_size]

    for i in range(num_global_batches):  # because of tagger: it looks forwards

        # cnn part
        sent_x = sent_x_train[i * sent_batch_size:(i + 1) * sent_batch_size]
        sent_y = sent_y_train[i * sent_batch_size:(i + 1) * sent_batch_size]

        if i % 100 == 0:
            print("batch number",i)

        yield sent_x, sent_y


sent_eval_batch_size = 64
#with train_graph.as_default():
#    saver = tf.train.Saver()

#print(len(train_words), len(sent_x_train), len(sent_y_train), len(tagger_train_words), len(train_labels_id))

with tf.Session(graph=train_graph) as sess:
    #iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())
    #train EVERYTHING
    for e in range(1, epochs + 1):
        BATCH = get_joint_batches(sent_x_train, sent_y_train)
        for x_2, y_2 in BATCH:
            sess.run(sent_train_op, feed_dict={sent_X: x_2, sent_Y: y_2})

            #iteration += 1


    #evaluate CNN
    sent_prediction = np.array([])
    i = 0
    while i * sent_eval_batch_size < len(sent_x_test):
        x_batch = sent_x_test[i * sent_eval_batch_size:(i + 1) * sent_eval_batch_size]
        y_batch = sent_y_test[i * sent_eval_batch_size:(i + 1) * sent_eval_batch_size]
        i += 1
        a = sess.run(sent_batch_prediction, feed_dict={sent_X: x_batch, sent_Y: y_batch})
        sent_prediction = np.append(sent_prediction, np.asarray(a))

    # Obtain label predictions by rounding predictions to int
    sent_prediction = [int(round(t)) for t in sent_prediction]

    # Use F1 metric:
    F1 = f1_score(y_true=sent_y_test, y_pred=sent_prediction, average=None)
    print("SENTIMENT F1 score: ", F1)

    sess.close()
