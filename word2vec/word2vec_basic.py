from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf

def maybe_download(filename, expected_bytes):
    """
    Download if file not present
    Args:
        filename: the name of the file to download
        expected_bytes: the exact size,in bytes, of the file to download
    Returns:
        filename:
    """
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url+filename,filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print("found and verified %s" %filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify %s. Can you get it with a browser?' %filename)
    return filename

#Read the data into a list of strings
def read_data(filename):
    """
    Extract the first file stored in a zip as a list of words
    Args:
        filename: the string name of the file to generate words from
    Returns:
        data: the list of words in the file
    """
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        # print(type(data))
        # print(data[0:25])
    return data

#Step 3: build the dictionary and replace words with UNK token
def build_dataset(words, vocabulary_size):
    """
    Build dataset of words and their counts
    Args:
        words: the whole
        vocabulary_size: the size of the vocabulary to store
    Returns
        data:
        count:
        dictionary:
        reverse_dictionary:
    """
    #count stores the count of each word in words
    count = [['UNK',-1]] #The count of 'UNK' is -1
    #add counter for vocabulary_size-1 words from the given words
    count.extend((collections.Counter(words).most_common(vocabulary_size -1)))
    dictionary = dict()
    #uncomment i to see how the dictionary is made
    # i = 0;
    #dictionary of word IDs
    for word,_ in count:
        # print(dictionary)
        # i+=1
        # if i < 10:
        #     print(len(dictionary))
        dictionary[word] =len(dictionary)

    data = list()
    unk_count = 0
    #word converted to wordID vector
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0 #dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

#Step3: generate training batch for skip-gram model
def generate_batch(batch_size, num_skips, skip_window):
    """
    Function to generate training for skip-gram model
    Args:
        batch_size:
        num_skips:
        skip_window:
    Returns:
        batch:
        labels:
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

def plot_with_labels(low_dim_embs, labels, filename='tsnt.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18,18)) #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                    xy = (x,y),
                    xytext = (5,2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')
    plt.savefig(filename)

if __name__ == '__main__':

    #STEP1: Download the data
    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download('text8.zip',31344016)
    #Read the data into a list of strings
    words = read_data(filename)
    print('Data size %d' %len(words))
    #STEP2: Build the dictionary and replace rare words with UNK token
    vocabulary_size = 50000
    data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
    del words #words not needed anymore so reduce memory consumption by deleting
    print('Most common words (+UNK) {}'.format(count[:5]))
    print('Sample data {}:{}'.format(data[:10], [reverse_dictionary[i] for i in data[:10]]))
    print('least common words {}'.format(count[len(count)-5:len(count)]))
    #STEP3: Generate training batch for skip-gram model
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]],'->', labels[i, 0], reverse_dictionary[labels[i, 0]])
    #STEP4: Build and train skip gram model
    batch_size = 128
    embedding_size = 128 #Dimension of the embedding vector
    skip_window = 1 #How many words to consider left and right
    num_skips = 2 #How many times to reuse an input to generate a label
    #We pick a random validation set to sample nearest neighbours. Here we limit the
    #validation samples to the words that have a low numeric ID, which by construction, are also the moost
    valid_size = 16 #Random set of words to evaluate similarity on
    valid_window = 100 #Only pick dev samples in the head of the distrubution
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 20  #Number of negative examples to sample
    # print(valid_examples)

    graph = tf.Graph()

    with graph.as_default():
        #Input data
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        #Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            #Look up embeddings for inputs
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size],-1.0,1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            #Construct the variables for the NCE loss

            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0/math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        #Compute the average NCE loss for the batch
        #tf.nce_loss automatically draws a new sample of the negative labels each
        #time we evaluate the loss.
        loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels, num_sampled, vocabulary_size))
        #Construct the SGD optimizer using a learning rate of 1.0
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        #Compute the cosine similarity between the minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_mean(tf.square(embeddings),1,keep_dims=True))
        normalized_embeddings = embeddings/norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        #Add variable initializer
        init = tf.initialize_all_variables()

    #STEP5: Begin training
    num_steps = 100001
    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them
        init.run()
        print('initialized ')
        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            #We perform one update by evaluating the optimizer op(including it in the list of returned values for session.run())
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step %2000 == 0:
                if step > 0:
                    average_loss /= 2000
                #The average loss is an estimate of the loss over the last 2000 batches
                print("Average loss at the step {}: {}".format(step, average_loss))
                #Reset average loss
                average_loss = 0

            #Note that this is expensive(~20% slowdown if computed every 500 steps)

            if step %10000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8 #Number of nearest neighbours
                    nearest = (-sim[i,:]).argsort()[1:top_k+1]
                    log_str = "Nearest to %s:" %(valid_word)
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," %(log_str, close_word)
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()

    #STEP6: Visualize the embeddings
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        tsne = TSNE(perplexity=30, n_components=2, init='pca',n_iter=5000)
        plot_only=500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
        labels = [reverse_dictionary[i] for i in xrange(plot_only)]
        plot_with_labels(low_dim_embs, labels)
        print("DONE")
    except ImportError:
        print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
