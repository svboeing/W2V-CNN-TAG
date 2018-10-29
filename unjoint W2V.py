import numpy as np
import tensorflow as tf
from collections import Counter
import random
# Set global parameters:
n_embedding = 200
num_global_batches = 5000
epochs = 15
sub_threshold = 1e-5


# Set w2v parameters:
n_sampled = 100
w2v_window_size = 6
w2v_test_batch_size = 10 ##### WHATS THERe

with open('text8') as f:
    text = f.read()

w2v_words = text.split(' ')
w2v_words = w2v_words[1:-int(round(0.05*len(w2v_words)))]
w2v_test = w2v_words[-int(round(0.05*len(w2v_words))):]
print(len(w2v_words), len(w2v_test))

# Sort word tokens by frequencies and save them:
word_counts = Counter(w2v_words)
counts = [count for count in word_counts.values()]
unique_sorted_words = [word for word in sorted(word_counts, key=lambda k: word_counts[k], reverse=True)]
#unique = set(unique_sorted_words)
unique_sorted_words = np.asarray(unique_sorted_words)
unique = set(unique_sorted_words[:5000])
#w2v_tokens = set(unique)

#np.save("/home/boeing/PycharmProjects/CNN/JOINT_sorted_words", unique_sorted_words)

#w2v_test = [word for word in w2v_test if word in unique]

# Set vocabs. id = 0 is reserved for 'unknown' word
vocab_to_int = {}
int_to_vocab = {}

for i, word in enumerate(unique_sorted_words[:5000]):
    vocab_to_int[word] = i+1
    int_to_vocab[i+1] = word

n_vocab = len(int_to_vocab) + 1



int_words = [vocab_to_int[word] for word in w2v_words if word in unique]
w2v_test = [vocab_to_int[word] for word in w2v_test if word in unique]
# Subsampling:
word_counts = Counter(int_words)
total_count = len(int_words)
freqs = {word: count / total_count for word, count in word_counts.items()}
p_drop = {word: 1 - np.sqrt(sub_threshold / freqs[word]) for word in word_counts}
w2v_train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]
w2v_batch_size = len(w2v_train_words) // num_global_batches
w2v_train_words = w2v_train_words[:num_global_batches * w2v_batch_size]
w2v_test_batch_number = len(w2v_test) // w2v_test_batch_size
w2v_test = w2v_test[:w2v_test_batch_size * w2v_test_batch_number]
# Making batches


def get_target(words, idx, window_size):
    ''' Get a list of words in a window around an index. '''
    
    R = np.random.randint(1, window_size+1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = set(words[start:idx] + words[idx+1:stop+1])
    
    return list(target_words)    


# Here's a function that returns batches for our network.
# The idea is that it grabs `batch_size` words from a words list. Then for each of those words,
# it gets the target words in the window. I haven't found a way to pass in a random number of target words and
# get it to work with the architecture, so I make one row per input-target pair.
# This is a generator function by the way, helps save memory.

def get_joint_batches(w2v_train_words):
    for i in range(num_global_batches):

        # w2v part:
        w2v_x, w2v_y = [], []
        w2v_batch = w2v_train_words[i * w2v_batch_size:(i + 1) * w2v_batch_size]
        for ii in range(len(w2v_batch)):
            batch_x = w2v_batch[ii]
            batch_y = get_target(w2v_batch, ii, w2v_window_size)
            w2v_y.extend(batch_y)
            w2v_x.extend([batch_x] * len(batch_y))


        yield w2v_x, w2v_y



X = tf.placeholder(tf.int32, [None], name='inputs')
Y = tf.placeholder(tf.int32, [None, None], name='labels')
test_Y = tf.placeholder(tf.int32, [None], name='test_inputs')

# Embedding
embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1, seed = 123), name = 'common_embs')

softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding))) # create softmax weight matrix here
softmax_b = tf.Variable(tf.zeros(n_vocab), name="softmax_bias") # create softmax biases here

    # Calculate the loss using negative sampling
loss = tf.nn.sampled_softmax_loss(
    weights=softmax_w,
    biases=softmax_b,
    labels=Y,
    inputs=tf.nn.embedding_lookup(embedding, X),
    num_sampled=n_sampled,
    num_classes=n_vocab)

cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer().minimize(cost)

transposed_sotfmax_w = tf.transpose(softmax_w)
logits = tf.nn.bias_add(tf.matmul(embedding, transposed_sotfmax_w), softmax_b) # TEST IS ATTENTED
labels_one_hot = tf.one_hot(test_Y, n_vocab) #added
test_loss = tf.nn.softmax_cross_entropy_with_logits_v2( #added
            labels=labels_one_hot,
            logits=tf.nn.embedding_lookup(logits, X)
)



norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))
normalized_embedding = embedding / norm



#with train_graph.as_default():
#    saver = tf.train.Saver()

with tf.Session() as sess:
    loss = 0
    sess.run(tf.global_variables_initializer())


    sess.graph.finalize()


    # Training all three nets:
    for e in range(1, epochs + 1):
        print("epoch number", e)
        BATCH = get_joint_batches(w2v_train_words)


        for x_1, y_1 in BATCH:

            sess.run(optimizer, feed_dict={X: x_1, Y: np.array(y_1)[:, None]})

    # Save trained session:
    #save_path = saver.save(sess, "checkpoints/model.ckpt")
    #save_path = saver.save(sess, "checkpoints/FUCKINGmodel.ckpt")
    #embed_mat = sess.run(W2V_net.normalized_embedding)
    #embed_mat = np.asarray(embed_mat)
    #np.save("/home/boeing/PycharmProjects/CNN/w2v_a_embedding", embed_mat)

    # Evaluate w2v:
    w2v_scores = []
    for i in range(w2v_test_batch_number):
        w2v_x, w2v_y = [], []
        w2v_batch = w2v_test[i * w2v_test_batch_size:(i + 1) * w2v_test_batch_size]
        for ii in range(len(w2v_batch)):
            batch_x = w2v_batch[ii]
            batch_y = get_target(w2v_batch, ii, w2v_window_size)
            w2v_y.extend(batch_y)
            w2v_x.extend([batch_x] * len(batch_y))
        w2v_test_loss = sess.run(test_loss, feed_dict={X: w2v_x, test_Y: np.array(w2v_y)})
        w2v_scores = np.append(w2v_scores, np.asarray(w2v_test_loss))
    print(w2v_scores.shape)
    w2v_score = np.mean(w2v_scores)
    print("WORD TO VEC mean loss score:", w2v_score)
    sess.close()