import tensorflow as tf

import csv
from tqdm import tqdm
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import numpy as np

def init_process(fin, fout):
    """
        converts the raw sentiment140 data into required format.
        Format has 3 category labels and then the tweet
        the seperator used is '|' and any use of the character '|' in tweets has ben replaced with a space
        This had to be done instead of multi char seperator as both python's csv module and tensorflow's csv decoder require single character seperator
        Args:
            fin: string filename of input file
            fout: string filename of output file
        Returns:
            None
        Creates:
            csv file named fout
    """
    outfile = open(fout, 'a')
    with open(fin, buffering=200000, encoding='latin-1') as f:
        csvreader = csv.reader(f, quotechar='"')
        for line in csvreader:
            #the tweet is in the last column
            #replace any " in the tweet with null
            line[-1] = line[-1].replace('"','')
            #replace any "|" in the tweet with space as "|" will be used as a delimiter later
            line[-1] = line[-1].replace("|","  ")
            #The category is in the first column
            initial_polarity = line[0]

#                As mentioned above, in the markdown, using one-hot vector does not work
#               However, I found later that the issue was that the record_defaults parameter in
#               tensorflow's decode_csv can't take a list
#               so we convert the one-hot list in the old csv to 3 columns showing the category as 1 or 0
            if initial_polarity == '0':
                initial_polarity = '1|0|0'
            elif initial_polarity == '2':
                initial_polarity = '0|1|0'
            elif initial_polarity == '4':
                initial_polarity = '0|0|1'

            tweet = line[-1]
            outline = initial_polarity +'|'+tweet+'\n'
            outfile.write(outline)

    outfile.close()

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def create_lexicon(fin):
    """
    Creates lexocon
    """
    lexicon = []
    with open(fin, 'r', buffering=100000, encoding='latin-1') as f:
        try:
            counter = 1
            content = ''
            for line in tqdm(f,total=file_len(fin)):
                counter += 1
                if (counter/2500.0).is_integer():
                    tweet = line.split('|')[3]
                    content += ' '+tweet
                    words = word_tokenize(content)
                    words = [lemmatizer.lemmatize(i) for i in words]
                    lexicon = list(set(lexicon + words))
                    # print(counter, len(lexicon))

        except Exception as e:
            print(str(e))

    with open('lexicon.pickle','wb') as f:
        pickle.dump(lexicon,f)

def read_to_vec(fin, lexicon):
    """generates a vector for a line(USE GENERATOR)
    this function is just for demonstration purposes, to show how the feature vector and category vector should be returned later in tensorflow.
    it works but it will be slower
    Args:
        fin: filename to read from
        lexicon: the lexicon, premade using training data and loaded from pickle file
    Returns:
        None
    Yields:
        labels: one hot labels from csv
        features: feature vector from csv
    """
    with open(fin, 'r') as f:
        csvreader = csv.reader(f,delimiter='|')
        for line in csvreader:
            current_words = word_tokenize(line[-1].lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            labels = list(int(val) for val in (line[0:3]))
            yield(labels,list(features)) #every time the function is called, return a label, features pair is returned


def read_record(filename_queue):
    """
    Outputs tensorflow ops to read one tweet and one set of labels
    Args:
        filename_queue: tensorflow queue that contains the files to be read
        lexicon: the lexicon, premade using training data and loaded from pickle file
        sess: the tensorflow session to run this in
    Returns:
        feature_op: tensorflow op that returns tweet from one record
        label_op: tensorflow op that returns one-hot label vector in numpy array of shape 3 from one record
    """
    #filename_queue = tf.train.string_input_producer([fin])

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    #default value of positive label
    cat1_default = tf.constant([0])
    #default value of neutral label
    cat2_default = tf.constant([1])
    #default value of negative label
    cat3_default = tf.constant([0])
    #default value of tweet
    tweet_default = tf.constant(['default tweet'])
    #combine the default values in a list
    record_defaults = [cat1_default,cat2_default,cat3_default, tweet_default]
    #get ops to read the values
    cat1, cat2, cat3, raw_tweet_op = tf.decode_csv(value, record_defaults=record_defaults,field_delim='|')
    #combine the labels into one list
    label_op = tf.stack([cat1,cat2,cat3])

    return raw_tweet_op, label_op


def get_vector(filename_queue,lexicon,sess,coord,threads):
    """
    Does preprocessing to convert raw_tweet returned from the tensorflow reader to vector
    Made for the batch reading but does not currently work
    Args:
        filename_queue: tensorflow filename_queue that feeds in the files to process
        lexicon: the lexicon, premade using training data and loaded from pickle file
    Returns:
        features: feature vector showing if a word is in the tweet
        label: labels showing the sentiment polarity
    """
    raw_tweet_op,label_op = read_record(filename_queue)
    with tf.Session() as sess:
        # sess.run(tf.initialize_local_variables())
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord = coord)
        sess.run(tf.initialize_local_variables())
        raw_tweet, label = sess.run([raw_tweet_op, label_op])
        # print(label)
        print(raw_tweet)
        coord.request_stop()
        coord.join(threads)

    # tflexicon = tf.constant(lexicon)
    tweet = str(raw_tweet,'utf-8')
    features = np.zeros(len(lexicon))
    for word in tweet:
        if word.lower() in lexicon:
            index_value = lexicon.index(word.lower())
            features[index_value] += 1
    # print(features)
    return list(features), label


def input_pipeline(filename_queue, batch_size, lexicon,sess,coord,threads,num_epochs=None):
    """
        Reads multiple lines of data and then creates a batch of size batch_size
        NOTE: does not currently work
        Args:
            filename_queue: the tensorflow queue from which to read the file
            batch_size: the size of the batch to output
            lexicon: the lexicon containing the words
            sess: the tensorflow session in which to run the operations
            coord: the coordinator used in the session
            threads: the threads used in the session
            num_epochs: the number of epochs to run this for
        Returns:
            example_batch: batch of example feature vectors
            label_batch: batch of labels
    """
    # filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    #get one preprosed example and one label
    example, label = get_vector(filename_queue,lexicon,sess, coord, threads)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)

    return example_batch, label_batch


import random
def shuffle_data(fin):
    """
    function to shuffle input data as it is ordered by default with all the negative tweets first so this is required to mix the tweet categories
    Args:
        fin: string filename of file to shuffle
    """
    with open(fin,'r') as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    with open('./data/shuffled_train_data.csv','w') as target:
        for _, line in data:
            target.write( line )


def main():
    """uncomment to make the csv file"""
    # init_process('./data/training.1600000.processed.noemoticon.csv','./data/train_data.csv')
    # init_process('./data/testdata.manual.2009.06.14.csv','./data/test_data.csv')
    with open('./lexicon.pickle','rb') as f:
        lexicon = pickle.load(f)

    """uncomment to test the generator function"""
    # vec_getter = read_to_vec('train_data.csv',lexicon)
    # for i in range(10):
    #     print(next(vec_getter))

    """uncomment to test the tensorflow version of the generator"""
    """we use test data here as the training data is ordered and it is difficult to see if the whole dataset is fetched"""
    # filename_queue = tf.train.string_input_producer(['test_data.csv'],num_epochs=2)
    # try:
    #     with tf.Session() as sess:
    #         # sess.run(tf.initialize_all_variables())
    #         sess.run(tf.initialize_local_variables())
    #         coord = tf.train.Coordinator()
    #         threads = tf.train.start_queue_runners(coord=coord)
    #
    #         feature_op, label_op = read_record(filename_queue,lexicon,sess)
    #         # feature_op, label_op = input_pipeline(filename_queue, 128, lexicon, sess)
    #         count = 0
    #         while not coord.should_stop():
    #             label = sess.run([label_op])
    #             print(label)
    #             count +=1
    #             print(count)
    # except tf.errors.OutOfRangeError:
    #     print("Done training, epoch reached")
    # finally:
    #     coord.request_stop()
    # coord.join(threads)

    """uncomment to test the batch generator"""
    # filename_queue = tf.train.string_input_producer(['test_data.csv'],num_epochs=1)
    # try:
    #     with tf.Session() as sess:
    #         sess.run(tf.initialize_all_variables())
    #         sess.run(tf.initialize_local_variables())
    #         coord = tf.train.Coordinator()
    #         threads = tf.train.start_queue_runners(coord=coord)
    #         feature_op, label_op = input_pipeline(filename_queue, 100, lexicon,sess,coord,threads,num_epochs=1)
    #         count = 0
    #         while not coord.should_stop():
    #             features = sess.run(feature_op)
    #             # print(features)
    #             print(str(features[0],'utf-8'))
    #             count +=1
    #             print(count)
    # except tf.errors.OutOfRangeError:
    #     print("Done training, epoch reached")
    # finally:
    #     coord.request_stop()
    # coord.join(threads)

    """uncomment to test the get_vector function"""
    # filename_queue = tf.train.string_input_producer(['test_data.csv'],num_epochs=None)
    # print(get_vector(filename_queue,lexicon))


    """uncomment to shuffle the input file and check 10 records"""
    shuffle_data('./data/train_data.csv')
    with open('./data/shuffled_train_data.csv','r') as f:
        for i in range(10):
            print(f.readline())

if __name__ == '__main__':
    """
        Read the comments and file and the main function to see how to use
    """
    # import time
    # start = time.time()
    main()
    # fin = time.time()
    # print(fin-start)
