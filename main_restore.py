import numpy as np
import tensorflow as tf
from collections import Counter
import random
from keras.datasets import imdb
from keras.preprocessing import sequence
from sklearn.metrics import f1_score

# Import our modules w/ net classes
import CNNmodule
import W2Vmodule
import TAGGERmodule

# Set global parameters:
n_embedding = 200
num_global_batches = 5000
epochs = 15
sub_threshold = 1e-5

# Set CNN parameters:
sent_max_features = 5000
sent_maxlen = 400
n_filters = 250
sent_kernel_size = 3
sent_hidden_dims = 250
sent_learning_rate = 0.003
sent_width = 3
sent_eval_batch_size = 64

# Set w2v parameters:
n_sampled = 100
w2v_window_size = 6
w2v_test_batch_size = 1000

# Set tagger parameters:
tagger_learning_rate = 0.01
n_hidden_1 = 256
n_hidden_2 = 64
num_input = 24
num_classes = 17
n_tag_emb = 30
n_suf_emb = 10

with open('text8') as f:
    text = f.read()

w2v_words = text.split(' ')
w2v_words = w2v_words[1:-int(round(0.05*len(w2v_words)))]
w2v_test = w2v_words[-int(round(0.05*len(w2v_words))):]

unique_sorted_words = np.load("/home/boeing/PycharmProjects/CNN/JOINT_sorted_words.npy")
unique = set(unique_sorted_words)

w2v_test = [word for word in w2v_test if word in unique]

# Set vocabs. id = 0 is reserved for 'unknown' word
vocab_to_int = {}
int_to_vocab = {}

for i, word in enumerate(unique_sorted_words):
    vocab_to_int[word] = i+1
    int_to_vocab[i+1] = word

int_words = [vocab_to_int[word] for word in w2v_words]
w2v_test = [vocab_to_int[word] for word in w2v_test]
# Subsampling:
word_counts = Counter(int_words)
total_count = len(int_words)
freqs = {word: count / total_count for word, count in word_counts.items()}
p_drop = {word: 1 - np.sqrt(sub_threshold / freqs[word]) for word in word_counts}
w2v_train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]
w2v_batch_size = len(w2v_train_words) // num_global_batches
#maybe gotta get test to full batches
w2v_test_batch_number = len(w2v_test) // w2v_test_batch_size
w2v_train_words = w2v_train_words[:num_global_batches * w2v_batch_size]
w2v_test = w2v_test[:w2v_test_batch_size * w2v_test_batch_number]
for i, word in enumerate(unique_sorted_words):
    vocab_to_int[word] = i+1
    int_to_vocab[i+1] = word

# Preprocess CNN (sentiment analysis) input:
(sent_train_words_raw, sent_train_labels), (sent_test_words_raw, sent_test_labels) = imdb.load_data(
    num_words=sent_max_features, index_from=3)

# IMDB word tokens are encoded with their indexes of frequency. Decode:
imdb_w_to_id = imdb.get_word_index()
imdb_w_to_id = {k: (v + 3) for k, v in imdb_w_to_id.items()}
imdb_w_to_id["<PAD>"] = 0
imdb_w_to_id["<START>"] = 1
imdb_w_to_id["<UNK>"] = 2

imdb_id_to_w = {value: key for key, value in imdb_w_to_id.items()}

sent_train_words, sent_test_words = [], []

for i in range(len(sent_train_words_raw)):
    sent_train_words.append([vocab_to_int[imdb_id_to_w[_id]] for _id in sent_train_words_raw[i]
                             if imdb_id_to_w[_id] in unique])
    sent_test_words.append([vocab_to_int[imdb_id_to_w[_id]] for _id in sent_test_words_raw[i]
                            if imdb_id_to_w[_id] in unique])

sent_train_words = sequence.pad_sequences(sent_train_words, maxlen=sent_maxlen)
sent_test_words = sequence.pad_sequences(sent_test_words, maxlen=sent_maxlen)
# Only full batches on training:
sent_batch_size = len(sent_train_words) // num_global_batches
sent_train_words = sent_train_words[:num_global_batches * sent_batch_size]
sent_train_labels = sent_train_labels[:num_global_batches * sent_batch_size]

# Preprocess tagger input:

tags_line = "ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X"
tags_list = tags_line.split(' ')
letters_line = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
letters = letters_line.split(' ')

word_id = vocab_to_int

tag_id = {}
pref_id = {}

for i, tag in enumerate(tags_list):
    tag_id[tag] = i + 1

Id = 1
for letter1 in letters:
    for letter2 in letters:
        pref_id[letter1.lower() + letter2.lower()] = Id
        Id += 1

# Extract words and their tags from file from universaltransitions.com:
tagged_words = []
tagged_words_tags = []
with open("en_pud-ud-test.conllu", "r") as f:
    for line in f:
        if line[0] == '#':
            continue
        splitted = line.split('\t')
        if len(splitted) <= 3:
            continue
        if splitted[1].lower() in unique and len(splitted[1]) > 1:
            tagged_words.append(splitted[1].lower())
            tagged_words_tags.append(splitted[3])

# Split corpus by 80/20:
tagger_train_amount = round(len(tagged_words) * 0.8) #????????????????????????????????????????????????

# TAGS number 12 and 14 are NOT IN THE DATASET. Tag number 6 is not in the test dataset.

tagger_train_words = tagged_words[:tagger_train_amount]
train_tags = tagged_words_tags[:tagger_train_amount]
tagger_test_words = tagged_words[tagger_train_amount:]
test_tags = tagged_words_tags[tagger_train_amount:]

tagger_train_labels = [tag_id[tag] for tag in train_tags]
tagger_test_labels = [tag_id[tag] for tag in test_tags]

tagger_all_labels = [tag_id[tag]-1 for tag in tagged_words_tags]
#tagger_all_labels = [t-1 for t in tagger_all_labels]

# Only full batches on training:
tagger_batch_size = len(tagger_train_words) // num_global_batches
tagger_train_words = tagger_train_words[:num_global_batches * tagger_batch_size]
tagger_train_labels = tagger_train_labels[:num_global_batches * tagger_batch_size]

# Length of vocab:
n_vocab = len(int_to_vocab) + 1

# Common embeddings variable:
embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1))

# Create instances of neural nets with parameters specified:
CNN_net = CNNmodule.CNN(embedding, n_embedding, sent_max_features, sent_maxlen, n_filters,
                 sent_kernel_size, sent_hidden_dims, sent_learning_rate, sent_width, sent_eval_batch_size)

W2V_net = W2Vmodule.W2V(embedding, n_vocab, n_embedding, sub_threshold, n_sampled, w2v_window_size)

TAGGER_net = TAGGERmodule.TAGGER(embedding, word_id, pref_id, tag_id, tagger_batch_size, n_embedding, tagger_learning_rate,
                 n_hidden_1, n_hidden_2, num_input, num_classes, n_tag_emb, n_suf_emb)


saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    # Evaluate w2v:
    w2v_scores = []
    for i in range(w2v_test_batch_number):
        w2v_x, w2v_y = [], []
        w2v_batch = w2v_test[i * w2v_test_batch_size:(i + 1) * w2v_test_batch_size]
        for ii in range(len(w2v_batch)):
            batch_x = w2v_batch[ii]
            batch_y = W2V_net.get_target(w2v_batch, ii)
            w2v_y.extend(batch_y)
            w2v_x.extend([batch_x] * len(batch_y))
        w2v_test_loss = sess.run(W2V_net.test_loss, feed_dict={W2V_net.X: w2v_x, W2V_net.test_Y: np.array(w2v_y)})
        w2v_scores = np.append(w2v_scores, np.asarray(w2v_test_loss))
    print(w2v_scores.shape)
    w2v_score = np.mean(w2v_scores)
    print("WORD TO VEC mean loss score:", w2v_score)


    # Evaluate tagger:
    #tagger_test_acc = np.array([], dtype="float32")
    tags_learned = np.array([], dtype="int32")
    for i in range(len(tagger_test_words)):
        x = TAGGER_net.form_minibatch(tagger_test_words, tags_learned, i, 1, word_id, pref_id)
        y = TAGGER_net.form_onehot_batch(tagger_test_labels, i, 1)


        logits = sess.run(TAGGER_net.test_logits,
                                         feed_dict={TAGGER_net.X: x, TAGGER_net.Y: y})
        logits = logits[0]

        label_learned = np.argmax(logits)
        tags_learned = np.append(tags_learned, label_learned)

    tagger_test_labels = np.asarray(tagger_test_labels, dtype = 'int32')-np.ones(len(tagger_test_labels), dtype = 'int32')

    for i in range(num_classes):
        print(i, i in tagger_test_labels, i in tags_learned, i in tagger_all_labels)
    score = np.asarray(f1_score(y_true=tagger_test_labels,
                                y_pred=tags_learned, average=None))
    score_weighted = f1_score(y_true=tagger_test_labels,
                                y_pred=tags_learned, average='weighted')
    print('TAGGER WEIGHTED F1 score:', score_weighted)

    present_labels = [label for label in range(num_classes) if label in tagger_test_labels and label in tags_learned]
    tagger_f1_dict = dict(zip(present_labels, score))
    print(tagger_f1_dict)
    # Evaluate CNN:
    sent_prediction = np.array([])
    i = 0
    while i * sent_eval_batch_size < len(sent_test_words):
        x = sent_test_words[i * sent_eval_batch_size:(i + 1) * sent_eval_batch_size]
        y = sent_test_labels[i * sent_eval_batch_size:(i + 1) * sent_eval_batch_size]
        i += 1
        a = sess.run(CNN_net.batch_prediction, feed_dict={CNN_net.X: x, CNN_net.Y: y})
        sent_prediction = np.append(sent_prediction, np.asarray(a))

    # Obtain label predictions by rounding predictions to int:
    sent_prediction = [int(round(t)) for t in sent_prediction]
    #print(sent_prediction, sent_test_labels)
    # Use F1 metric:
    F1 = f1_score(y_true=sent_test_labels, y_pred=sent_prediction, average=None)
    with open("cnn_accs.txt", "a") as f:
        f.write(str(F1))
    print("SENTIMENT F1 score:", F1)
    sess.close()