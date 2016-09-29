import tensorflow as tf

import csv

import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import numpy as np

def init_process(fin, fout):
    outfile = open(fout, 'a')
    with open(fin, buffering=200000, encoding='latin-1') as f:
        csvreader = csv.reader(f, quotechar='"')
        for line in csvreader:
            line[-1] = line[-1].replace('"','')
            line[-1] = line[-1].replace("|","  ")
            initial_polarity = line[0]
            # print(line[0])
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


def read_to_vec(fin, lexicon):
    """generates a vector for a line(USE GENERATOR)
    this function is just for demonstration purposes, to show how the feature vector and category vector is returned later in tensorflow
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
    NOTE: To be completed,
    PROBLEM: I can't figure out how to do both the read and the conversion while still in the tensorflow op
    so I am currently reading from decode_csv, then calling the same, then converting and then retrning the converted feature vec
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
    cat1_default = tf.constant([0])
    cat2_default = tf.constant([1])
    cat3_default = tf.constant([0])
    tweet_default = tf.constant(['default tweet'])
    record_defaults = [cat1_default,cat2_default,cat3_default, tweet_default]
    cat1, cat2, cat3, raw_tweet_op = tf.decode_csv(value, record_defaults=record_defaults,field_delim='|')

    #Run the op to get the string features
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord)
    # raw_tweet = sess.run(raw_tweet_op)
    # tweet = str(raw_tweet,'utf-8')
    # features = np.zeros(len(lexicon))
    # for word in tweet:
    #     if word.lower() in lexicon:
    #         index_value = lexicon.index(word.lower())
    #         features[index_value] +=1
    # feature_op = tf.pack(list(features))
    label_op = tf.pack([cat1,cat2,cat3])
    # coord.request_stop()
    # coord.join(threads)
    return raw_tweet_op, label_op


def get_vector(filename_queue,lexicon):
    """
    Does preprocessing to convert raw_tweet returned from the tensorflow reader to vector
    Args:
        filename_queue:
        lexicon:
    Returns:
        feature_vec:
    """
    raw_tweet_op,label_op = read_record(filename_queue)
    with tf.Session() as sess:
        sess.run(tf.initialize_local_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)

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
    return list(features), label


def input_pipeline(filename_queue, batch_size, lexicon,num_epochs=None):
    # filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    example, label = get_vector(filename_queue,lexicon)
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

def main():
    """uncomment to make the csv file"""
    # init_process('training.1600000.processed.noemoticon.csv','train_data.csv')
    # init_process('testdata.manual.2009.06.14.csv','test_data.csv')
    with open('../lexicon.pickle','rb') as f:
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
    filename_queue = tf.train.string_input_producer(['test_data.csv'],num_epochs=1)
    try:
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            sess.run(tf.initialize_local_variables())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            feature_op, label_op = input_pipeline(filename_queue, 100, lexicon,num_epochs=1)
            count = 0
            while not coord.should_stop():
                features = sess.run(feature_op)
                # print(features)
                print(str(features[0],'utf-8'))
                count +=1
                print(count)
    except tf.errors.OutOfRangeError:
        print("Done training, epoch reached")
    finally:
        coord.request_stop()
    coord.join(threads)

    """uncomment to test the get_vector function"""
    # filename_queue = tf.train.string_input_producer(['test_data.csv'],num_epochs=None)
    # print(get_vector(filename_queue,lexicon))

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    fin = time.time()
    print(fin-start)
