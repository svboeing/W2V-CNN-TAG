import numpy as np
import tensorflow as tf

# Set parameters
learning_rate = 0.01
epochs = 15
batch_size = 64
n_hidden_1 = 256
n_hidden_2 = 64
num_input = 24
num_classes = 17

# Make lists of tags and english letters:
tags_line = "ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X"
tags_list = tags_line.split(' ')
letters_line = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
letters = letters_line.split(' ')

# Extract words and its embeddings from fasttext's output, skipping first line:
skip = False
words = []
vecs = []
with open("fasttext150", "r") as f:
    for line in f:
        if skip:
            info = line.strip().split(' ')
            words.append(info[0])
            c = []
            for i in range(1, len(info)):
                c.append(float(info[i]))
            vecs.append(c)
        else:
            skip = True

vecs = np.asarray(vecs, dtype='float32')

# Create dictionaries for words(text8), tags, suf, pref. Id = 0 is reserved. Prefix dictionary is the same as suffix.
word_id = {}
tag_id = {}
pref_id = {}

# Embedding for word with id = 0("unknown word") is all zeros:
vecs_lookup = np.concatenate([np.zeros((1, 100), dtype='float32'), vecs], axis=0)
Id = 1
for word in words:
    if word not in word_id:
        word_id[word] = Id
        Id += 1

Id = 1
for tag in tags_list:
    tag_id[tag] = Id
    Id += 1

Id = 1
for letter1 in letters:
    for letter2 in letters:
        pref_id[letter1.lower()+letter2.lower()] = Id
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
        if splitted[1].lower() in words and len(splitted[1]) > 1:
            tagged_words.append(splitted[1].lower())
            tagged_words_tags.append(splitted[3])

# Split corpora:
train_amount = round(len(tagged_words) * 0.8)
train_words = tagged_words[:train_amount]
train_tags = tagged_words_tags[:train_amount]
test_words = tagged_words[train_amount:]
test_tags = tagged_words_tags[train_amount:]
print("Words total: ", len(tagged_words), "Train total: ", len(train_tags), "Test total: ", len(test_tags))

train_labels_id = [tag_id[tag] for tag in train_tags]
test_labels_id = [tag_id[tag] for tag in test_tags]

word_embs = tf.constant(vecs_lookup, dtype=tf.float32)
print(tf.shape(word_embs))
suf_init = tf.random_normal(shape=[len(pref_id)+1, 10], dtype="float32")
suf = tf.Variable(suf_init, name='suffixes')
pref = tf.Variable(suf_init, name='prefixes')

tag_init = tf.random_normal(shape=[len(tag_id)+1, 30], dtype="float32")
tags = tf.Variable(tag_init, name='tags')


def get_suf(some_word):
    return some_word[-2:]


def get_pref(some_word):
    return some_word[:2]


# Form supervector from words, tags, suffixes and prefixes, looking backwards and forwards from pos:
def supervector_maker(words_list, tags_list, pos):

    if pos in range(3, len(words_list)-3):
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


# Embedding lookup for words, tags, suffixes, prefixes from minibatch:
def form_feed(minibatch, batch_size):
    words_id_block = minibatch[:, :7]
    tags_id_block = minibatch[:, 7:10]
    pref_id_block = minibatch[:, 10:17]
    suf_id_block = minibatch[:, 17:24]
    print(minibatch.shape)
    words_lookup = tf.reshape(tf.nn.embedding_lookup(word_embs, words_id_block), (batch_size, -1))
    tag_lookup = tf.reshape(tf.nn.embedding_lookup(tags, tags_id_block), (batch_size, -1))
    suf_lookup = tf.reshape(tf.nn.embedding_lookup(suf, suf_id_block), (batch_size, -1))
    pref_lookup = tf.reshape(tf.nn.embedding_lookup(pref, pref_id_block), (batch_size, -1))
    feed = tf.concat([words_lookup, tag_lookup, pref_lookup, suf_lookup], 1)
    return feed


def form_onehot_batch(tags_learned, i, batch_size):
    matrix = []
    for counter in range(i, i+batch_size):
        hollow = np.zeros(17)
        hollow[tags_learned[counter]-1] = 1
        matrix.append(hollow)
    matrix = np.asarray(matrix).astype('int32')
    return matrix


# Store layers weight & bias
xavier_init = tf.contrib.layers.xavier_initializer()
weights = {
    'h1': tf.Variable(xavier_init([930, n_hidden_1])),
    'h2': tf.Variable(xavier_init([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(xavier_init([n_hidden_2, 17]))
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'out': tf.Variable(tf.zeros([17]))
}

X = tf.placeholder(tf.int32, [None, 24])
Y = tf.placeholder(tf.int32, [None, num_classes])


def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    x_feed = form_feed(x, batch_size)
    layer_1 = tf.add(tf.matmul(x_feed, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 64 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Construct model
logits = neural_net(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

output_labels = tf.argmax(logits, 1)


# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
sess = tf.InteractiveSession()

# Run the initializer
sess.run(init)

for step in range(1, epochs + 1):
    # for step in range(1, 2):
    for i in range(len(train_words) - batch_size):
        x_batch = form_minibatch(train_words, train_labels_id, i, batch_size)
        y_batch = form_onehot_batch(train_labels_id, i, batch_size)
        loss_batch, acc_batch, _ = sess.run([loss_op, accuracy, train_op], feed_dict={X: x_batch, Y: y_batch})
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
    print("Step " + str(step) + ", Minibatch Loss= " + \
          "{:.4f}".format(loss_batch) + ", Training Accuracy= " + \
          "{:.3f}".format(acc_batch))

print("Optimization Finished!")


# Evaluation section: change batch_size to 1 because now tags are unknown:
batch_size = 1
test_acc = np.array([], dtype="float32")

X_test = tf.placeholder(tf.int32, [1, 24])
Y_test = tf.placeholder(tf.int32, [1, num_classes])
print(X_test, Y_test)

# Evaluate model
logits = neural_net(X_test)
output_labels = tf.argmax(logits, 1)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_test, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


tags_learned = np.array([], dtype="int32")

for i in range(len(test_words) - 1):
    x = form_minibatch(test_words, tags_learned, i, batch_size)
    y = form_onehot_batch(test_labels_id, i, batch_size)

    acc = sess.run(accuracy, feed_dict={X_test: x, Y_test: y})
    test_acc = np.append(test_acc, acc)
    label_learned = sess.run(output_labels, feed_dict={X_test: x, Y_test: y})
    tags_learned = np.append(tags_learned, label_learned)


print("Testing Accuracy =", np.mean(test_acc))
sess.close()
