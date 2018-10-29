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


# Set tagger params

tagger_learning_rate = 0.01
# epochs = 5
# batch_size = 64
n_hidden_1 = 256
n_hidden_2 = 64
num_input = 24
num_classes = 17
n_tag_emb = 30
n_suf_emb = 10

unique_sorted_words = np.load("/home/boeing/PycharmProjects/CNN/JOINT_sorted_words.npy")
unique = set(unique_sorted_words)

vocab_to_int = {}
int_to_vocab = {}

for i, word in enumerate(unique_sorted_words):
    vocab_to_int[word] = i+1 #!!!!!!!!!!!!!!!! # NOW 0 IS RESERVED ONCE AGAIN - SAME AS TAGGER_S
    int_to_vocab[i+1] = word #!!!!!!!!!!!!!!!!!!!


# TAGGER PART

tags_line = "ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X"
tags_list = tags_line.split(' ')
letters_line = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
letters = letters_line.split(' ')


# Create dictionaries for words(text8), tags, suf, pref. Id = 0 is reserved. Prefix dictionary is the same as suffix.
# word_id = {} # WE DO ALREADY HAVE NUMBERED WORDS vocab_to_int - but it starts with the:0.
word_id = vocab_to_int

tag_id = {}
pref_id = {}

# Embedding for word with id = 0("unknown word") is all zeros:
# vecs_lookup = np.concatenate([np.zeros((1, 100), dtype='float32'), vecs], axis=0)
# Id = 1
# for word in words:
#    if word not in word_id:
#        word_id[word] = Id
#        Id += 1


for i, tag in enumerate(tags_list):
    tag_id[tag] = i + 1

Id = 1
for letter1 in letters:
    for letter2 in letters:
        pref_id[letter1.lower() + letter2.lower()] = Id
        Id += 1

# Extract words and their tags from file from universaltransitions.com
tagged_words = []
tagged_words_tags = []
with open("en_pud-ud-test.conllu", "r") as f:
    for line in f:
        if line[0] == '#':
            continue
        splitted = line.split('\t')
        if len(splitted) <= 3:
            continue
        # Simply throw away words that have no embs
        if splitted[1].lower() in unique and len(splitted[1]) > 1: # SO THAT WE ONLY USE WORDS FROM TEXT8 - EVERYWHERE. IN TRAIN AND IN TESTING
            tagged_words.append(splitted[1].lower())
            tagged_words_tags.append(splitted[3])

# Split corpora:
tagger_train_amount = round(len(tagged_words) * 0.8)
tagger_train_words = tagged_words[:tagger_train_amount]
train_tags = tagged_words_tags[:tagger_train_amount]
tagger_test_words = tagged_words[tagger_train_amount:]
test_tags = tagged_words_tags[tagger_train_amount:]
# print("Words total: ", len(tagged_words), "Train total: ", len(train_tags), "Test total: ", len(test_tags))

train_labels_id = [tag_id[tag] for tag in train_tags]
test_labels_id = [tag_id[tag] for tag in test_tags]


def get_suf(some_word):
    return some_word[-2:]


def get_pref(some_word):
    return some_word[:2]


# Form supervector from words, tags, suffixes and prefixes, looking backwards and forwards from pos:
def supervector_maker(words_list, tags_list, pos):
    if pos in range(3, len(words_list) - 3):
        supervector = np.array([word_id[words_list[pos - 3]], word_id[words_list[pos - 2]],
                                word_id[words_list[pos - 1]], word_id[words_list[pos]], word_id[words_list[pos + 1]],
                                word_id[words_list[pos + 2]], word_id[words_list[pos + 3]], tags_list[pos - 3],
                                tags_list[pos - 2], tags_list[pos - 1], pref_id[get_pref(words_list[pos - 3])],
                                pref_id[get_pref(words_list[pos - 2])], pref_id[get_pref(words_list[pos - 1])],
                                pref_id[get_pref(words_list[pos])], pref_id[get_pref(words_list[pos + 1])],
                                pref_id[get_pref(words_list[pos + 2])], pref_id[get_pref(words_list[pos + 3])],
                                pref_id[get_suf(words_list[pos - 3])], pref_id[get_suf(words_list[pos - 2])],
                                pref_id[get_suf(words_list[pos - 1])], pref_id[get_suf(words_list[pos])],
                                pref_id[get_suf(words_list[pos + 1])], pref_id[get_suf(words_list[pos + 2])],
                                pref_id[get_suf(words_list[pos + 3])]])

    if pos == 0:
        supervector = np.array([0, 0, 0, word_id[words_list[pos]],
                                word_id[words_list[pos + 1]], word_id[words_list[pos + 2]],
                                word_id[words_list[pos + 3]], 0, 0, 0, 0, 0, 0,
                                pref_id[get_pref(words_list[pos])], pref_id[get_pref(words_list[pos + 1])],
                                pref_id[get_pref(words_list[pos + 2])], pref_id[get_pref(words_list[pos + 3])], 0, 0, 0,
                                pref_id[get_suf(words_list[pos])], pref_id[get_suf(words_list[pos + 1])],
                                pref_id[get_suf(words_list[pos + 2])], pref_id[get_suf(words_list[pos + 3])]])

    if pos == 1:
        supervector = np.array([0, 0, word_id[words_list[pos - 1]], word_id[words_list[pos]],
                                word_id[words_list[pos + 1]], word_id[words_list[pos + 2]],
                                word_id[words_list[pos + 3]], 0, 0, tags_list[pos - 1], 0, 0,
                                pref_id[get_pref(words_list[pos - 1])], pref_id[get_pref(words_list[pos])],
                                pref_id[get_pref(words_list[pos + 1])], pref_id[get_pref(words_list[pos + 2])],
                                pref_id[get_pref(words_list[pos + 3])], 0, 0, pref_id[get_suf(words_list[pos - 1])],
                                pref_id[get_suf(words_list[pos])], pref_id[get_suf(words_list[pos + 1])],
                                pref_id[get_suf(words_list[pos + 2])], pref_id[get_suf(words_list[pos + 3])]])

    if pos == 2:
        supervector = np.array([0, word_id[words_list[pos - 2]], word_id[words_list[pos - 1]], word_id[words_list[pos]],
                                word_id[words_list[pos + 1]], word_id[words_list[pos + 2]],
                                word_id[words_list[pos + 3]], 0, tags_list[pos - 2], tags_list[pos - 1],
                                0, pref_id[get_pref(words_list[pos - 2])], pref_id[get_pref(words_list[pos - 1])],
                                pref_id[get_pref(words_list[pos])], pref_id[get_pref(words_list[pos + 1])],
                                pref_id[get_pref(words_list[pos + 2])], pref_id[get_pref(words_list[pos + 3])],
                                0, pref_id[get_suf(words_list[pos - 2])], pref_id[get_suf(words_list[pos - 1])],
                                pref_id[get_suf(words_list[pos])], pref_id[get_suf(words_list[pos + 1])],
                                pref_id[get_suf(words_list[pos + 2])], pref_id[get_suf(words_list[pos + 3])]])

    if pos == len(words_list) - 1:
        supervector = np.array([word_id[words_list[pos - 3]], word_id[words_list[pos - 2]],
                                word_id[words_list[pos - 1]], word_id[words_list[pos]], 0, 0, 0, tags_list[pos - 3],
                                tags_list[pos - 2], tags_list[pos - 1], pref_id[get_pref(words_list[pos - 3])],
                                pref_id[get_pref(words_list[pos - 2])], pref_id[get_pref(words_list[pos - 1])],
                                pref_id[get_pref(words_list[pos])], 0, 0, 0, pref_id[get_suf(words_list[pos - 3])],
                                pref_id[get_suf(words_list[pos - 2])], pref_id[get_suf(words_list[pos - 1])],
                                pref_id[get_suf(words_list[pos])], 0, 0, 0])

    if pos == len(words_list) - 2:
        supervector = np.array([word_id[words_list[pos - 3]], word_id[words_list[pos - 2]],
                                word_id[words_list[pos - 1]], word_id[words_list[pos]], word_id[words_list[pos + 1]], 0,
                                0, tags_list[pos - 3], tags_list[pos - 2], tags_list[pos - 1],
                                pref_id[get_pref(words_list[pos - 3])], pref_id[get_pref(words_list[pos - 2])],
                                pref_id[get_pref(words_list[pos - 1])], pref_id[get_pref(words_list[pos])],
                                pref_id[get_pref(words_list[pos + 1])], 0, 0, pref_id[get_suf(words_list[pos - 3])],
                                pref_id[get_suf(words_list[pos - 2])], pref_id[get_suf(words_list[pos - 1])],
                                pref_id[get_suf(words_list[pos])], pref_id[get_suf(words_list[pos + 1])], 0, 0])

    if pos == len(words_list) - 3:
        supervector = np.array([word_id[words_list[pos - 3]], word_id[words_list[pos - 2]],
                                word_id[words_list[pos - 1]], word_id[words_list[pos]], word_id[words_list[pos + 1]],
                                word_id[words_list[pos + 2]], 0, tags_list[pos - 3], tags_list[pos - 2],
                                tags_list[pos - 1], pref_id[get_pref(words_list[pos - 3])],
                                pref_id[get_pref(words_list[pos - 2])], pref_id[get_pref(words_list[pos - 1])],
                                pref_id[get_pref(words_list[pos])], pref_id[get_pref(words_list[pos + 1])],
                                pref_id[get_pref(words_list[pos + 2])], 0, pref_id[get_suf(words_list[pos - 3])],
                                pref_id[get_suf(words_list[pos - 2])], pref_id[get_suf(words_list[pos - 1])],
                                pref_id[get_suf(words_list[pos])], pref_id[get_suf(words_list[pos + 1])],
                                pref_id[get_suf(words_list[pos + 2])], 0])

    return supervector


# Form minibatch from supervectors, given pos and batch size:
def form_minibatch(words_list, tags_list, pos, batch_size):
    matrix = []
    for counter in range(pos, pos + batch_size):
        line = supervector_maker(words_list, tags_list, counter)
        matrix.append(line)
    matrix = np.asarray(matrix).astype('int32')
    return matrix


# COMMON EMBEDDINGS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
train_graph = tf.Graph()
n_vocab = len(int_to_vocab) + 1

with train_graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1)) #cuz 0 is reserved!!!!!!!!!!

    xavier_init = tf.contrib.layers.xavier_initializer()

    # tagger graph nodes
    weights = {
        'h1': tf.Variable(xavier_init([7 * n_embedding + 3 * n_tag_emb + 14 * n_suf_emb, n_hidden_1])),
    # whhaaaaaaaaaaat
        'h2': tf.Variable(xavier_init([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(xavier_init([n_hidden_2, num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1])),
        'b2': tf.Variable(tf.zeros([n_hidden_2])),
        'out': tf.Variable(tf.zeros([num_classes]))
    }
    suf_init = tf.random_normal(shape=[len(pref_id) + 1, n_suf_emb], dtype="float32")
    suf = tf.Variable(suf_init, name='suffixes')
    pref = tf.Variable(suf_init, name='prefixes')

    tag_init = tf.random_normal(shape=[len(tag_id) + 1, n_tag_emb], dtype="float32")
    tags = tf.Variable(tag_init, name='tags')

    tagger_X = tf.placeholder(tf.int32, [None, 24]) #24 comes from supervector
    tagger_Y = tf.placeholder(tf.int32, [None, num_classes]) #num_classes is really len(tag_id)
    #tagger_a = tf.Variable(tf.random_uniform([n_embedding,], -1, 1))
    #tagger_a_embedding = tf.multiply(embedding, tagger_a)

tagger_batch_size = len(tagger_train_words) // num_global_batches
# Embedding lookup for words, tags, suffixes, prefixes from minibatch:
def form_feed(minibatch, batch_size):  # beforehand batch_size was passed
    words_id_block = minibatch[:, :7]
    tags_id_block = minibatch[:, 7:10]
    pref_id_block = minibatch[:, 10:17]
    suf_id_block = minibatch[:, 17:24]
    # WHERE DOES THIS LOOKUP WATCH?????
    #print(minibatch.shape)
    #batch_size = minibatch.shape[0]
    #batch_size = tagger_batch_size
    words_lookup = tf.reshape(tf.nn.embedding_lookup(embedding, words_id_block), (batch_size, -1))
    tag_lookup = tf.reshape(tf.nn.embedding_lookup(tags, tags_id_block), (batch_size, -1))
    suf_lookup = tf.reshape(tf.nn.embedding_lookup(suf, suf_id_block), (batch_size, -1))
    pref_lookup = tf.reshape(tf.nn.embedding_lookup(pref, pref_id_block), (batch_size, -1))
    feed = tf.concat([words_lookup, tag_lookup, pref_lookup, suf_lookup], 1)
    return feed


def form_onehot_batch(tags_learned, i, batch_size):
    matrix = []
    for counter in range(i, i + batch_size):
        hollow = np.zeros(17)
        hollow[tags_learned[counter] - 1] = 1
        matrix.append(hollow)
    matrix = np.asarray(matrix).astype('int32')
    return matrix


def tagger_neural_net(x, batch_size):
    # Hidden fully connected layer with 256 neurons
    x_feed = form_feed(x, batch_size)
    layer_1 = tf.add(tf.matmul(x_feed, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 64 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# COMMON


with train_graph.as_default():

    # Construct model
    tagger_logits = tagger_neural_net(tagger_X, tagger_batch_size)
    tagger_output_labels = tf.argmax(tagger_logits, 1)
    # Define loss and optimizer
    tagger_loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=tagger_logits, labels=tagger_Y))
    tagger_optimizer = tf.train.AdamOptimizer(learning_rate=tagger_learning_rate)
    tagger_train_op = tagger_optimizer.minimize(tagger_loss_op)



    test_tagger_logits = tagger_neural_net(tagger_X, 1)
    tagger_test_output_labels = tf.argmax(test_tagger_logits, 1)
    correct_pred = tf.equal(tagger_test_output_labels, tf.argmax(tagger_Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



    # Evaluate model (with test logits, for dropout to be disabled)
    #tagger_correct_pred = tf.equal(tf.argmax(tagger_logits, 1), tf.argmax(tagger_Y, 1))
    #tagger_accuracy = tf.reduce_mean(tf.cast(tagger_correct_pred, tf.float32))



def get_joint_batches(tagger_train_words, train_labels_id):

    tagger_train_words = tagger_train_words[:num_global_batches * tagger_batch_size]
    train_labels_id = train_labels_id[:num_global_batches * tagger_batch_size]

    for i in range(num_global_batches):  # because of tagger: it looks forwards
        # v2w part


        tagger_x_batch = form_minibatch(tagger_train_words, train_labels_id, i * tagger_batch_size, tagger_batch_size)
        tagger_y_batch = form_onehot_batch(train_labels_id, i * tagger_batch_size, tagger_batch_size)
        if i % 100 == 0:
            print("batch number",i)

        yield tagger_x_batch, tagger_y_batch



#with train_graph.as_default():
    #saver = tf.train.Saver()

print(len(tagger_train_words), len(train_labels_id))

with tf.Session(graph=train_graph) as sess:
    #iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())
    #train EVERYTHING
    for e in range(1, epochs + 1):
        BATCH = get_joint_batches(tagger_train_words, train_labels_id)
        for x_3, y_3 in BATCH:

            sess.run(tagger_train_op, feed_dict={tagger_X: x_3, tagger_Y: y_3})
            #iteration += 1

    test_acc = np.array([], dtype="float32")

    tags_learned = np.array([], dtype="int32")

    # print(tagger_test_words[:50])

    for i in range(len(tagger_test_words) - 1):
        x = form_minibatch(tagger_test_words, tags_learned, i, 1)
        y = form_onehot_batch(test_labels_id, i, 1)
        # print(x.shape, y.shape, x, y, form_feed(x, 1))
        acc = sess.run(accuracy, feed_dict={tagger_X: x, tagger_Y: y})
        test_acc = np.append(test_acc, acc)
        label_learned = sess.run(tagger_output_labels, feed_dict={tagger_X: x, tagger_Y: y})
        tags_learned = np.append(tags_learned, label_learned)

    print("AVERAGE TAGGER TEST ACCURACY =", np.mean(test_acc))
    sess.close()


