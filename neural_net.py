from data_read import read_record
import tensorflow as tf
import pickle



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
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
    init_op  = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    with open('../lexicon.pickle','rb') as f:
        lexicon = pickle.load(f)

    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        batch_x_op, batch_y_op = input_pipeline(['train_data.csv'], batch_size, lexicon, sess, num_epochs=1)
        try:
            feature_op, label_op = read_record(filename_queue,lexicon,sess)
            while not coord.should_stop():
                batch_x, batch_y = sess.run([batch_x_op, batch_y_op])
                print(batch_y)
                _,c = sess.run([optimizer,cost])
                print(c)
        except tf.errors.OutOfRangeError:
            print("Done training, epoch reached")
        finally:
            coord.join(threads)
train_neural_network(x)
