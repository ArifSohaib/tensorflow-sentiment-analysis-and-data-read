#Introduction
This tutorial uses the same technique as sentdex from youtube from this video: https://www.youtube.com/watch?v=JeamFbHhmDo
with a written tutorial here: https://pythonprogramming.net/data-size-example-tensorflow-deep-learning-tutorial/

My version adds tensorflow's built-in reader instead of manual iteration used in the original.

The reason for this is that for loops in python are known to be slow and manual iteration has to use them.
tensorflow, however, does this in C++ like all of its graph computation so theoretically, it should be much faster.
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

#The main difference
