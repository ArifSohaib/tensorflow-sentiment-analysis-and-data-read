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
