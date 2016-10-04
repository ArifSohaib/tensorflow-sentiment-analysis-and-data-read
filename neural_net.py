from data_read import read_record
import tensorflow as tf
import pickle
import numpy as np


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 3

batch_size=128
total_batches = int(1600000/batch_size)
hm_epochs = 15

x = tf.placeholder(tf.float32,shape=[None, 2638])
y = tf.placeholder(tf.float32)


hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([2638, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}

def neural_network_model(data):
    # hidden_1_layer, hidden_2_layer, hidden_3_layer, output_layer = initialize_weights_and_biases(lexicon)
    l1 = tf.nn.relu(tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias']))
    l2 = tf.nn.relu(tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias']))
    l3 = tf.nn.relu(tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias']))
    output = tf.add(tf.matmul(l3, output_layer['weight']),output_layer['bias'])
    return output

saver = tf.train.Saver()
tf_log = 'tf.log'

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    filename_queue = tf.train.string_input_producer(['shuffled_train_data.csv'],num_epochs=hm_epochs)
    with open('../lexicon.pickle', 'rb') as f:
        lexicon = pickle.load(f)
    tweet_op, label_op = read_record(filename_queue)
    batch_counter = 0
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())
        try:
            epoch =int(open(tf_log,'r').read().split('\n')[-2])+1
            print("Starting: Epoch %d" % epoch)
        except:
            epoch = 1

        # while epoch < hm_epochs:
        try:
            if epoch != 1:
                saver.restore(sess, "model.ckpt")
            epoch_loss=1

            batch_x = []
            batch_y = []
            batches_run = 0
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)
            while not coord.should_stop():
                raw_tweet, label = sess.run([tweet_op,label_op])
                tweet = str(raw_tweet,'utf-8')
                features = np.zeros(len(lexicon))
                for word in tweet:
                    if word.lower() in lexicon:
                        index_value = lexicon.index(word.lower())
                        features[index_value] += 1
                        batch_x.append(list(features))
                        batch_y.append(label)
                if len(batch_x)>=batch_size:
                    _,c = sess.run([optimizer, cost], feed_dict={x:np.array(batch_x), y:np.array(batch_y)})
                    epoch_loss += c
                    batch_x = []
                    batch_y = []
                    batches_run += 1
                    print('sample label: {}'.format(label))
                    print('Batch run: {}/{} | Epoch: {} | Batch Loss: {}'.format(batches_run,total_batches, epoch, c))
                if batches_run == total_batches: #meaning one epoch completed
                    saver.save(sess, 'model.ckpt')
                    print('Epoch: {}, completed out of {}, Loss = {}'.format(epoch, hm_epochs, epoch_loss))
                    with open(tf_log, 'a') as f:
                        f.write(str(epoch)+'\n')
                        epoch +=1
                    batches_run=0 #reset the counter
        except tf.errors.OutOfRangeError:
            print("Done training, epoch reached")
        finally:
            coord.request_stop()
            threads.join(coord)

train_neural_network(x)

def test_neural_network():
    prediction = neural_network_model(x)
    filename_queue = tf.train.string_input_producer(['test_data.csv'],num_epochs=1)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    feature_sets = []
    labels = []
    counter = 0
    with open('../lexicon.pickle','rb') as f:
        lexicon = pickle.load(f)
    tweet_op, label_op = read_record(filename_queue)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())
        # for epoch in range(hm_epochs):
        try:
            # print(saver.latest_checkpoint())
            saver.restore(sess,"model.ckpt")
            print('restored network')
        except Exception as e:
            print(str(e))
            # epoch_loss = 0
        try:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)
            print(type(threads))
            while not coord.should_stop():
                tweet_raw, label = sess.run([tweet_op,label_op])
                # print(tweet_raw)
                tweet = str(tweet_raw, 'utf-8')
                # print(tweet)
                features = np.zeros(len(lexicon))
                for word in tweet:
                    if word.lower() in lexicon:
                        index_value = lexicon.index(word.lower())
                        features[index_value] +=1
                feature_sets.append(features)
                labels.append(label)

        except tf.errors.OutOfRangeError:
            test_x = np.array(feature_sets)
            test_y = np.array(labels)
            print(test_x.shape)
            print(test_y.shape)
            # correct_example = sess.run(correct, feed_dict={x:feature_sets,y:labels})
            # print(correct_example)
            print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))
            print("Epoch complete")
        finally:
            coord.request_stop()
            # threads.join(coord)

print('testing net')
test_neural_network()
