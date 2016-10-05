#import the read_record function from data_read file
from data_read import read_record

#the main imports
import tensorflow as tf
import pickle
import numpy as np

#set number of hidden nodes in each layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
#numner of classes. In our case it is positive, negative and neutral
n_classes = 3

batch_size=128
#there are 1600000 lines in the training data
total_batches = int(1600000/batch_size)
#total number of epochs(epoch is number of times the whole data set is seen )
hm_epochs = 5
#placeholders for the input and output
x = tf.placeholder(tf.float32,shape=[None, 2638])
y = tf.placeholder(tf.float32)

#Define the hidden layers' weights and biases
#the 2638 is the number of words in the lexicon created using the training data
hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([2638, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}
#define the output layers weights and biases
output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}

#The saver object
#this can only be made after declaring some tensorflow variables.
#in our case, those are the layer weights and biases above
saver = tf.train.Saver()
#log file to store number of epochs completed
tf_log = 'tf.log'

def neural_network_model(data):
    """
    Function to define the neural network model
    In this version it is just a feedforward multi layer preceptrion
    Args:
        data: tensorflow placeholder containing the feature vector/s
    Returns:
        output: Output tensor built after the network
    """
    #all of the layers follow the standard M * X + b function where M is the weights and b is the bias
    #then the hidden layers are passed through the relu function
    l1 = tf.nn.relu(tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias']))
    l2 = tf.nn.relu(tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias']))
    l3 = tf.nn.relu(tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias']))
    output = tf.add(tf.matmul(l3, output_layer['weight']),output_layer['bias'])
    return output


def train_neural_network(x):
    """
    Function to train the neural network
    Args:
        x: the tensorflow placeholder for the input feature vector
    """
    #the feed forward tensorflow operation
    prediction = neural_network_model(x)
    #cost operation
    #tf.nn.softmax_cross_entropy_with_logits is used instead of tf.nn.softmax when we have one-hot vectors as both the labels and the output from the network
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,100000, 0.96, staircase=True)
    # Passing global_step to minimize() will increment it at each step.
    learning_step = (tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step))

    #Define the optimizer and the value to minimize
    #NOTE Following line is disabled to check decaying learning_rate
    # optimizer = tf.train.AdamOptimizer().minimize(cost)
    #queue of input files
    filename_queue = tf.train.string_input_producer(['shuffled_train_data.csv'],num_epochs=hm_epochs)
    #get the lexicon built and saved previously from the training data
    with open('../lexicon.pickle', 'rb') as f:
        lexicon = pickle.load(f)
    #tensorflow operations to get one tweet and one set of one-hot labels from the input file
    tweet_op, label_op = read_record(filename_queue)

    #All tensorflow operations need to be run from a session
    with tf.Session() as sess:
        #the tensorflow variables(eg; weights and biases declared earlier) have to be initialized using this function
        #otherwise they are just tensors describing those variables
        sess.run(tf.initialize_all_variables())
        #the epoch is counted internally in tensorflow. This counter needs to be initialized using this function
        sess.run(tf.initialize_local_variables())
        #try to get the epoch number from the log file
        try:
            epoch =int(open(tf_log,'r').read().split('\n')[-2])+1
            print("Starting: Epoch %d" % epoch)
        #if the file does not open, assume we are on epoch 1
        except:
            epoch = 1

        try:
            #if some epochs have already been run, then restore from checkpoint file called 'model.ckpt' using the saver
            if epoch != 1:
                saver.restore(sess, "model.ckpt")
            epoch_loss=1
            #the batch of input vectors
            batch_x = []
            #the batch of labels
            batch_y = []
            #to keep track of the number of batches run in one epoch
            batches_run = 0
            #the following two lines are essencial for using tensorflow's data reader from the file queue (filename_queue) above
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)
            #the should_stop function keeps running the following code until the epochs in the filename_queue are completed
            #the epoch counter is internal which is why we used the tf.initialize_local_variables earlier
            while not coord.should_stop():
                #run the operation to get one tweer as a binary string and one set of one-hot labels
                raw_tweet, label = sess.run([tweet_op,label_op])
                #convert the binary tweet to utf-8(python3's default string encoding)
                tweet = str(raw_tweet,'utf-8')
                #a numpy array of zeros representing, each representing one word in the lexicon
                features = np.zeros(len(lexicon))
                #if any word in the tweet is in the lexicon, we increment its count in the feature vector
                for word in tweet:
                    if word.lower() in lexicon:
                        index_value = lexicon.index(word.lower())
                        features[index_value] += 1
                #add the tweet's feature vector to the batch
                batch_x.append(list(features))
                batch_y.append(label)
                #when a batch is filled
                if len(batch_x)>=batch_size:
                    #run the optimizer operation and the cost operation using the batch
                    _,c = sess.run([learning_step, cost], feed_dict={x:np.array(batch_x), y:np.array(batch_y)})
                    #increment the epoch loss by the cost c of the batch
                    epoch_loss += c
                    #empty the batch
                    batch_x = []
                    batch_y = []
                    #increment the batch counter
                    batches_run += 1
                    #print the label of the last example in the batch for debugging/sanity check
                    # print('sample label: {}'.format(label))
                    #show the batch loss and the number of batches run in the given epoch
                    print('Batch run: {}/{} | Epoch: {} | Batch Loss: {}'.format(batches_run,total_batches, epoch, c))
                if batches_run == total_batches: #meaning one epoch completed
                    saver.save(sess, 'model.ckpt') #save the variables at each epoch

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

# train_neural_network(x)

def test_neural_network():
    """
    Function to test the neural network

    """
    #tensorflow op to get prediction
    prediction = neural_network_model(x)
    #tensorflow queue for the input file
    filename_queue = tf.train.string_input_producer(['test_data.csv'],num_epochs=1)
    #tensorflow op to check if prediction is correct
    #the argmax function gives the index of the maximum value on the given axis. In this case axis 1
    #axis 1 meaning the row axis here
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    #tensorflow operation to get the accuracy of the classifier
    #it converts the result of the previous op from boolean to float32 and then takes the mean over the result
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    #lists for features and labels for the input
    feature_sets = []
    labels = []

    #get the lexicon
    with open('../lexicon.pickle','rb') as f:
        lexicon = pickle.load(f)
    #tensorflow operation to get one tweet and one set of labels
    tweet_op, label_op = read_record(filename_queue)
    with tf.Session() as sess:
        #initialize the saved variables
        sess.run(tf.initialize_all_variables())
        #this is required to initialize the internal epoch counter used by the filename_queue
        sess.run(tf.initialize_local_variables())
        #get the trained model or show an error
        try:
            # print(saver.latest_checkpoint())
            saver.restore(sess,"model.ckpt")
            print('restored network')
        except Exception as e:
            print(str(e))

        try:
            #tensorflow coordinator to coordinate the threads
            coord = tf.train.Coordinator()
            #tensorflow queue runner required to get input from the filename_queue
            threads = tf.train.start_queue_runners(coord = coord)
            #run the following until the number of epochs specified in filename_queue are completed
            while not coord.should_stop():
                #get one tweet as binary string and one set of labels
                tweet_raw, label = sess.run([tweet_op,label_op])
                #if you want to see the tweet, uncomment the line below
                # print(tweet_raw)
                #convert the tweet to utf-8, python3's default string encoding
                tweet = str(tweet_raw, 'utf-8')
                #uncomment below to see the converted tweet
                # print(tweet)
                #initialize a numpy array of zeros for each word in the lexicon
                features = np.zeros(len(lexicon))
                #if a word in the tweet is in the lexicon, increment the count of that word in the feature vector
                for word in tweet:
                    if word.lower() in lexicon:
                        index_value = lexicon.index(word.lower())
                        features[index_value] +=1
                #append the feature vector and label to the input list
                feature_sets.append(features)
                labels.append(label)
        #since the test file is small, the whole file is processed in the above lines
        #when this is done we get an OutOfRangeError meaning the epochs specified have been completed
        #in this case, we only need 1 epoch and when it is complete we classify the test data
        except tf.errors.OutOfRangeError:
            #convert the lists to numpy arrays
            test_x = np.array(feature_sets)
            test_y = np.array(labels)
            #for debugging pruposes, the shape of the numpy array can be checked
            #the first dimension should equal the number of lines in the test csv
            # print(test_x.shape)
            # print(test_y.shape)
            #uncomment below if you want to see the actual boolean array that shows which examples were correctly classified
            # correct_example = sess.run(correct, feed_dict={x:feature_sets,y:labels})
            # print(correct_example)
            # print(correct_example.shape)
            # get the accuracy by running the accuracy operation
            # accuracy.eval is just a different way of saying sess.run(accuracy, feed_dict={x:test_x, y:test_y})
            print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))
            print("Epoch complete")
        #stop the coordinator and join the threads
        finally:
            coord.request_stop()
            # threads.join(coord)

print('testing net')
test_neural_network()
