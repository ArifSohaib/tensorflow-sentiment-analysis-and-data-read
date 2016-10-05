#Introduction
This tutorial uses the same technique as sentdex from youtube from this video: https://www.youtube.com/watch?v=JeamFbHhmDo
with a written tutorial here: https://pythonprogramming.net/data-size-example-tensorflow-deep-learning-tutorial/

My version adds tensorflow's built-in reader instead of manual iteration used in the original.

The reason for this is that for loops in python are known to be slow and manual iteration has to use them.\n
Tensorflow, however, does this in C++ like all of its graph computation so theoretically, it should be much faster.
I am making this both for my own testing and because, while tensorflow has many tutorial and a lot of documentation, I personally haven't found any that explains its data reading beyond feed_dict well.

#The dataset
The dataset used here is the sentiment 140 dataset which can be found here:
http://help.sentiment140.com/for-students/
and the direct download link is
http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
with a mirror link https://docs.google.com/file/d/0B04GJPshIjmPRnZManQwWEdTZjg/edit
You need to get this data before running the network

#File descriptions
I am using two main files here:
#####data_read.py:
Contains functions for reading in and preprocessing the data and creating the lexicon of words
#####neural_net.py:
Contains the main neural network model and testing function

#Reading from tensorflow
There are a few other differences between this and sentdex's version but the main one is using tensorflow's built-in data reader.
Here is some description of it:\n
Firstly you need to put the input files in a queue using this:
filename_queue = tf.train.string_input_producer(['filenames'], num_epochs=None, shuffle=True)
file name have to be in a list and multiple files can be passed in. This is useful for distributed environments as well as for datasets that use multiple files like the IMDB review dataset
which has files called neg and pos for the two categories.

The queue needs to be declared before running the tensorflow session(tf.Session() or tf.InteractiveSession()).
If a number is passed in num_epochs then before running the reader operation, which I will mention later, you need to initialize local variables using tf.initialize_local_variables()
as tensorflow counts the epochs using a local variable which are not initialized when callling tf.initialize_all_variables() as is normally done.

tensorflow has many readers. In our case, we are reading a csv line by line so we use tf.TextLineReader() which is declared like this:
reader = tf.TextLineReader()
and a line reading operation is declared like this
key, value = reader.read(filename_queue) where key is the line being read and is assigned by tensorflow automatically
After this, we need to describe the data format in the csv and give default values for each column. These are also used to find the data type and convert automatically from raw string
for example if a line contains a numeric value we use:\n
```python
val_default = tf.constant([0])
```
or if it contains a string we use:\n
```python
str_default = tf.constant(['default string'])
```
note that lists can not be used in default values so\n
```python
lst_dafault = tf.constant([[0][1][2]]) returns an error
```
for simplicity, all the default values can be put in a list before being passed to the csv decoder\n
```python
record_defaults = [default1, default2, default3]
```
where each default represents one column separated by a given one character delimiter. Note that it has to be a one character delimiter due to the internal C operation requiring one char delimiter.
The decoding can be done using the following function\n
```python
tf.decode_csv(value, record_defaults=record_defaults,field_delim='|')
```
here, each default value in the record_defaults list results in a tensorflow operation. For example if we have 3 default values we have\n
```python
default1_op, default2_op, default3_op = tf.decode_csv(value, record_defaults=record_defaults,field_delim='|')
```
where all the returned values are tensorflow operations that return the mentioned column when run\n
Before running the operations from the tensorflow csv decoder, we need to use a queue runner which should be coordinated by a coordinator and this needs to be done within the tensorflow session\n
```python
coord = tf.train.Coordinator()\n
threads = tf.train.start_queue_runners(coord=coord)\n
```
after this, each time the operations from the decoder are run, they return their respective column from one line of the input files
