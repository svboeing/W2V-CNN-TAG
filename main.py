import numpy as np
import tensorflow as tf
from collections import Counter
import random
from keras.datasets import imdb
from keras.preprocessing import sequence
from sklearn.metrics import f1_score

# Import our modules w/ net classes
#import CNNmodule
import W2Vmodule
import TAGGERmodule
import ENCODERmodule

# Set global parameters:
n_embedding = 200
num_global_batches = 5000
epochs = 15 #15
sub_threshold = 1e-5
attention_size = 2

# Set WDISC params:
wdisc_batch_size = 64
# Set CNN parameters:
#sent_max_features = 5000
#sent_maxlen = 400
#n_filters = 250
#sent_kernel_size = 3
#sent_hidden_dims = 250
##sent_learning_rate = 0.003
#sent_width = 3
#sent_eval_batch_size = 64

# Set w2v parameters:
n_sampled = 100
w2v_window_size = 6
w2v_test_batch_size = 10 ##### WHATS THERE

# Set tagger parameters:
tagger_learning_rate = 0.01
n_hidden_1 = 256
n_hidden_2 = 64
num_input = 24
num_classes = 17
n_tag_emb = 30
n_suf_emb = 10



# Preprocess w2v input:
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

np.save("/home/boeing/PycharmProjects/CNN/JOINT_sorted_words", unique_sorted_words)

#w2v_test = [word for word in w2v_test if word in unique]

# Set vocabs. id = 0 is reserved for 'unknown' word
vocab_to_int = {}
int_to_vocab = {}

for i, word in enumerate(unique_sorted_words[:5000]):
    vocab_to_int[word] = i+1
    int_to_vocab[i+1] = word

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


# Preprocess CNN (sentiment analysis) input:
#(sent_train_words_raw, sent_train_labels), (sent_test_words_raw, sent_test_labels) = imdb.load_data(
#    num_words=sent_max_features, index_from=3)

# IMDB word tokens are encoded with their indexes of frequency. Decode:
#imdb_w_to_id = imdb.get_word_index()
#imdb_w_to_id = {k: (v + 3) for k, v in imdb_w_to_id.items()}
#imdb_w_to_id["<PAD>"] = 0
#imdb_w_to_id["<START>"] = 1
#imdb_w_to_id["<UNK>"] = 2

#imdb_id_to_w = {value: key for key, value in imdb_w_to_id.items()}

#sent_train_words, sent_test_words = [], []

#for i in range(len(sent_train_words_raw)):
#    sent_train_words.append([vocab_to_int[imdb_id_to_w[_id]] for _id in sent_train_words_raw[i]
#                             if imdb_id_to_w[_id] in unique])
#    sent_test_words.append([vocab_to_int[imdb_id_to_w[_id]] for _id in sent_test_words_raw[i]
 #                           if imdb_id_to_w[_id] in unique])

#sent_train_words = sequence.pad_sequences(sent_train_words, maxlen=sent_maxlen)
#sent_test_words = sequence.pad_sequences(sent_test_words, maxlen=sent_maxlen)
# Only full batches on training:
#sent_batch_size = len(sent_train_words) // num_global_batches
#sent_train_words = sent_train_words[:num_global_batches * sent_batch_size]
#sent_train_labels = sent_train_labels[:num_global_batches * sent_batch_size]

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
tagger_train_amount = round(len(tagged_words) * 0.8)
tagger_train_words = tagged_words[:tagger_train_amount]
train_tags = tagged_words_tags[:tagger_train_amount]
tagger_test_words = tagged_words[tagger_train_amount:]
test_tags = tagged_words_tags[tagger_train_amount:]

tagger_train_labels = [tag_id[tag] for tag in train_tags]
tagger_test_labels = [tag_id[tag] for tag in test_tags]

# Only full batches on training:
tagger_batch_size = len(tagger_train_words) // num_global_batches
tagger_train_words = tagger_train_words[:num_global_batches * tagger_batch_size]
tagger_train_labels = tagger_train_labels[:num_global_batches * tagger_batch_size]

# Length of vocab:
n_vocab = len(int_to_vocab) + 1

# Common embeddings variable:
#embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1, seed = 123), name = 'common_embs')

# Create instances of neural nets with parameters specified:

#PERC_net = PERCmodule.PERC(embedding, n_embedding, perc_n_hidden, perc_learning_rate, perc_n_sampled)

#better pass whole PERC_net?
#CNN_net = CNNmodule.CNN(embedding, PERC_net.cost, n_embedding, n_vocab, attention_size, sent_max_features, sent_maxlen, n_filters,
#                 sent_kernel_size, sent_hidden_dims, sent_learning_rate, sent_width, sent_eval_batch_size) #0
# Ha -Ha!!!! WORKS TILL HERE

def diff_loss(A, B):
    return tf.norm(tf.matmul(tf.transpose(tf.to_float(A)), tf.to_float(B)))**2

W2V_net = W2Vmodule.W2V(n_vocab, n_embedding, sub_threshold, n_sampled, w2v_window_size) #1

TAGGER_net = TAGGERmodule.TAGGER(word_id, n_vocab, pref_id, tag_id, tagger_batch_size, n_embedding,
                 tagger_learning_rate, n_hidden_1,n_hidden_2, num_classes, n_tag_emb, n_suf_emb) #2

W2V_priv_enc = ENCODERmodule.PRIVATE_ENCODER(W2V_net.own_embedding, n_embedding, n_hidden = 256, n_reduced = 100)

TAGGER_priv_enc = ENCODERmodule.PRIVATE_ENCODER(TAGGER_net.own_embedding, n_embedding, n_hidden = 256, n_reduced = 100)

COMMON_enc = ENCODERmodule.COMMON_ENCODER(W2V_net.own_embedding, TAGGER_net.own_embedding, n_embedding, n_hidden = 256, n_reduced = 100)

WAS_dist = ENCODERmodule.WDISC(COMMON_enc.encoded_data_1, COMMON_enc.encoded_data_2, n_reduced = 100, n_hidden=256, batch_size=64, LAMBDA=10)
# TILL HERE TODO check out whats wrong with taggers f1

COMMON_dec = ENCODERmodule.DECODER(tf.concat([COMMON_enc.encoded_data_1, W2V_priv_enc.encoded_data], 1),
                                   tf.concat([COMMON_enc.encoded_data_2, TAGGER_priv_enc.encoded_data], 1),
                                   tf.nn.embedding_lookup(W2V_net.own_embedding, COMMON_enc.X),
                                   tf.nn.embedding_lookup(TAGGER_net.own_embedding, COMMON_enc.X), n_reduced=100, n_embedding=200, n_hidden = 256)

COMMON_dec_train = tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.5,
            beta2=0.9
        ).minimize(
            COMMON_dec.loss,
            var_list=(COMMON_dec.weights['h1'], COMMON_dec.weights['h2'], COMMON_dec.weights['h3'], COMMON_dec.weights['h4'],
                      COMMON_dec.biases['b1'], COMMON_dec.biases['b2'], COMMON_dec.biases['b3'], COMMON_dec.biases['b4'],
                      COMMON_enc.weights['h1'], COMMON_enc.weights['h2'], COMMON_enc.weights['h3'], COMMON_enc.weights['h4'],
                      COMMON_enc.biases['b1'], COMMON_enc.biases['b2'], COMMON_enc.biases['b3'], COMMON_enc.biases['b4'],
                      W2V_priv_enc.weights['h1'], W2V_priv_enc.weights['h2'], W2V_priv_enc.weights['h3'], W2V_priv_enc.weights['h4'],
                      W2V_priv_enc.biases['b1'], W2V_priv_enc.biases['b2'], W2V_priv_enc.biases['b3'], W2V_priv_enc.biases['b4'],
                      TAGGER_priv_enc.weights['h1'], TAGGER_priv_enc.weights['h2'], TAGGER_priv_enc.weights['h3'], TAGGER_priv_enc.weights['h4'],
                      TAGGER_priv_enc.biases['b1'], TAGGER_priv_enc.biases['b2'], TAGGER_priv_enc.biases['b3'], TAGGER_priv_enc.biases['b4']
                      )
        )


# Wasserstein's distance similarity loss:
common_encoder_sim = -tf.reduce_mean(WAS_dist.wdisc_1) + tf.reduce_mean(WAS_dist.wdisc_2)
common_encoder_sim_train = tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.5,
            beta2=0.9
        ).minimize(
            common_encoder_sim,
            var_list=(COMMON_enc.weights['h1'], COMMON_enc.weights['h2'], COMMON_enc.weights['h3'], COMMON_enc.weights['h4'],
                      COMMON_enc.biases['b1'], COMMON_enc.biases['b2'], COMMON_enc.biases['b3'], COMMON_enc.biases['b4'])
        )
#Orthogonality difference constraint:
w2v_comm_priv_diff = diff_loss(COMMON_enc.encoded_data_1, W2V_priv_enc.encoded_data)
tagger_comm_priv_diff = diff_loss(COMMON_enc.encoded_data_2, TAGGER_priv_enc.encoded_data)
w2v_comm_priv_diff_train =  tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.5,
            beta2=0.9
        ).minimize(
            w2v_comm_priv_diff,
            var_list=(COMMON_enc.weights['h1'], COMMON_enc.weights['h2'], COMMON_enc.weights['h3'], COMMON_enc.weights['h4'],
                      COMMON_enc.biases['b1'], COMMON_enc.biases['b2'], COMMON_enc.biases['b3'], COMMON_enc.biases['b4'],
                      W2V_priv_enc.weights['h1'], W2V_priv_enc.weights['h2'], W2V_priv_enc.weights['h3'], W2V_priv_enc.weights['h4'],
                      W2V_priv_enc.biases['b1'], W2V_priv_enc.biases['b2'], W2V_priv_enc.biases['b3'], W2V_priv_enc.biases['b4']
                      )
        )
tagger_comm_priv_diff_train = tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.5,
            beta2=0.9
        ).minimize(
            tagger_comm_priv_diff,
            var_list=(COMMON_enc.weights['h1'], COMMON_enc.weights['h2'], COMMON_enc.weights['h3'], COMMON_enc.weights['h4'],
                      COMMON_enc.biases['b1'], COMMON_enc.biases['b2'], COMMON_enc.biases['b3'], COMMON_enc.biases['b4'],
                      TAGGER_priv_enc.weights['h1'], TAGGER_priv_enc.weights['h2'], TAGGER_priv_enc.weights['h3'], TAGGER_priv_enc.weights['h4'],
                      TAGGER_priv_enc.biases['b1'], TAGGER_priv_enc.biases['b2'], TAGGER_priv_enc.biases['b3'], TAGGER_priv_enc.biases['b4']
                      )
        )
def get_joint_batches(w2v_train_words, tagger_train_words, tagger_train_labels): #sent_train_words, sent_train_labels,
    for i in range(num_global_batches):

        # w2v part:
        w2v_x, w2v_y = [], []
        w2v_batch = w2v_train_words[i * w2v_batch_size:(i + 1) * w2v_batch_size]
        for ii in range(len(w2v_batch)):
            batch_x = w2v_batch[ii]
            batch_y = W2V_net.get_target(w2v_batch, ii)
            w2v_y.extend(batch_y)
            w2v_x.extend([batch_x] * len(batch_y))

        # cnn part:
        #sent_x = sent_train_words[i * sent_batch_size:(i + 1) * sent_batch_size]
        #sent_y = sent_train_labels[i * sent_batch_size:(i + 1) * sent_batch_size]

        # tagger part:
        tagger_x = TAGGER_net.form_minibatch(tagger_train_words, tagger_train_labels, i * tagger_batch_size,
                                             tagger_batch_size, word_id, pref_id)
        tagger_y = TAGGER_net.form_onehot_batch(tagger_train_labels, i * tagger_batch_size, tagger_batch_size)
        if i % 100 == 0:
            print("batch number",i)
        batch_words = set(w2v_batch) & set(
            [word_id[word] for word in tagger_train_words[i*tagger_batch_size : (i+1)*tagger_batch_size]])

        yield w2v_x, w2v_y, tagger_x, tagger_y, batch_words
saver = tf.train.Saver()

# TODO ask kostya about passing losses - how does it work?

with tf.Session() as sess:
    loss = 0
    sess.run(tf.global_variables_initializer())




    #testtest = [i for i in range(1, 65)]
    #print(len(testtest))
    #_, __ = sess.run((0.1 * WAS_dist.gradient_penalty, WAS_dist.disc_cost), feed_dict={COMMON_enc.X: testtest})
    #print(_, __)




    sess.graph.finalize()
    # Training all three nets:
    for e in range(1, epochs + 1):
        print("epoch number", e)
        BATCH = get_joint_batches(w2v_train_words,  tagger_train_words, tagger_train_labels) #sent_train_words, sent_train_labels

        counter = 0
        common_processed_words = set()
        for w2v_x, w2v_y, tagger_x, tagger_y, batch_words in BATCH:


            sess.run((W2V_net.train, W2V_net.train), feed_dict={W2V_net.X: w2v_x, W2V_net.Y: np.array(w2v_y)[:, None]})
            ##sess.run((CNN_net.attention_train,CNN_net.own_attention_train), feed_dict={CNN_net.X: x_2, CNN_net.Y: y_2})
            sess.run((TAGGER_net.train, TAGGER_net.train), feed_dict={TAGGER_net.X: tagger_x, TAGGER_net.Y: tagger_y})

            common_processed_words.update(batch_words)
            if len(common_processed_words) >= wdisc_batch_size:
                for i in range(1,10):
                    wdisc_batch = random.sample(common_processed_words, wdisc_batch_size)
                    sess.run((w2v_comm_priv_diff_train, tagger_comm_priv_diff_train, common_encoder_sim_train, WAS_dist.train_op, COMMON_dec_train), feed_dict={
                        COMMON_enc.X: wdisc_batch, W2V_priv_enc.X: wdisc_batch, TAGGER_priv_enc.X: wdisc_batch})
            #A, B, C = sess.run((sent_dist, w2v_dist, tagger_dist), feed_dict={w: w_})

            #batch_labels = np.argmin(np.array([A,B,C]), axis = 0)
            #one_hot_batch_labels = np.zeros([len(batch_labels), 3])
            #for i in range(len(batch_labels)):
            #    one_hot_batch_labels[i][batch_labels[i]] = 1
            #batch_true_labels_dict = dict(zip(w, batch_labels))



            #sess.run((PERC_net.train,W2V_net.adv_train), feed_dict={PERC_net.X: w_, PERC_net.Y: batch_labels[:, None]})

            counter += 1
            if counter % 500 == 0:
                _ = sess.run((W2V_net.cost), feed_dict={W2V_net.X: w2v_x, W2V_net.Y: np.array(w2v_y)[:, None]})
                print("w2v", _)
                _ = sess.run((TAGGER_net.loss), feed_dict={TAGGER_net.X: tagger_x, TAGGER_net.Y: tagger_y})
                print("tagger", _)
                if len(common_processed_words) >= wdisc_batch_size:
                    wdisc_batch = random.sample(common_processed_words, wdisc_batch_size)
                    a_, b_, c_, d_, e_ = sess.run((WAS_dist.disc_cost, common_encoder_sim, w2v_comm_priv_diff, tagger_comm_priv_diff, COMMON_dec.loss), feed_dict={
                        COMMON_enc.X: wdisc_batch, W2V_priv_enc.X: wdisc_batch, TAGGER_priv_enc.X: wdisc_batch})
                    print("Was. dist:", a_, "Common enc cost:", b_, "w2v orth", c_, "tag orth", d_, "dec", e_)
    # Save trained session:
    #save_path = saver.save(sess, "checkpoints/model.ckpt")
    save_path = saver.save(sess, "checkpoints/FUCKINGmodel.ckpt")
    embed_mat = sess.run(W2V_net.normalized_embedding)
    embed_mat = np.asarray(embed_mat)
    np.save("/home/boeing/PycharmProjects/CNN/w2v_a_embedding", embed_mat)

    # Evaluate w2v:
    w2v_scores = []
    for i in range(w2v_test_batch_number):
        if i % 100 == 0:
            print('w2v test batch number', i)
        w2v_x, w2v_y = [], []
        w2v_batch = w2v_test[i * w2v_test_batch_size:(i + 1) * w2v_test_batch_size]
        for ii in range(len(w2v_batch)):
            batch_x = w2v_batch[ii]
            batch_y = W2V_net.get_target(w2v_batch, ii)
            w2v_y.extend(batch_y)
            w2v_x.extend([batch_x] * len(batch_y))
        w2v_test_loss = sess.run(W2V_net.test_loss, feed_dict={W2V_net.X: w2v_x, W2V_net.test_Y: np.array(w2v_y)}) #[:, None] - check in restore and unjoint w2v
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

        #accuracy = sess.run(TAGGER_net.accuracy, feed_dict={TAGGER_net.X: x, TAGGER_net.Y: y})
        #tagger_test_acc = np.append(tagger_test_acc, accuracy)

        logits = sess.run(TAGGER_net.test_logits,
                          feed_dict={TAGGER_net.X: x, TAGGER_net.Y: y})
        logits = logits[0]

        label_learned = np.argmax(logits)
        tags_learned = np.append(tags_learned, label_learned)

    #with open("tagger_accs.txt", "a") as f:
    #    f.write(str(np.mean(tagger_test_acc)))
    #print("average tagger test accuracy =", np.mean(tagger_test_acc))
    tagger_test_labels = np.asarray(tagger_test_labels, dtype='int32') - np.ones(len(tagger_test_labels), dtype='int32')


    score = np.asarray(f1_score(y_true=tagger_test_labels,
                                y_pred=tags_learned, average=None))
    score_weighted = f1_score(y_true=tagger_test_labels,
                              y_pred=tags_learned, average='weighted')
    print('TAGGER WEIGHTED F1 score:', score_weighted)
    present_labels = [label for label in range(num_classes) if label in tagger_test_labels and label in tags_learned]
    tagger_f1_dict = dict(zip(present_labels, score))
    print(tagger_f1_dict)

    #print(score)


    # Evaluate CNN:
    #sent_prediction = np.array([])
    #i = 0
    #while i * sent_eval_batch_size < len(sent_test_words):
    #    x = sent_test_words[i * sent_eval_batch_size:(i + 1) * sent_eval_batch_size]
    #    y = sent_test_labels[i * sent_eval_batch_size:(i + 1) * sent_eval_batch_size]
    #    i += 1
    #    a = sess.run(CNN_net.batch_prediction, feed_dict={CNN_net.X: x, CNN_net.Y: y})
    #    sent_prediction = np.append(sent_prediction, np.asarray(a))

    # Obtain label predictions by rounding predictions to int:
    #sent_prediction = [int(round(t)) for t in sent_prediction]

    # Use F1 metric:
    #F1 = f1_score(y_true=sent_test_labels, y_pred=sent_prediction, average=None)
    #with open("cnn_accs.txt", "a") as f:
    #    f.write(str(F1))
    #print("SENTIMENT F1 score: ", F1)
    sess.close()