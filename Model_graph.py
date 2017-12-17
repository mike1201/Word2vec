from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import zipfile
import numpy as np
from six.moves import urllib
import tensorflow as tf
import data_preparation

# 2
np.random.seed(1)
tf.set_random_seed(1)

vocabulary_size = 500000
batch_size = 128
embedding_size = 128
skip_window = 2
num_skips = 4
num_sampled = 64
valid_size = 16

# 2-1
X_inputs = tf.placeholder(tf.int32, shape=[batch_size]) # X
Y_inputs = tf.placeholder(tf.int32, shape=[batch_size, 1]) # Y
valid_examples = np.random.choice(100, 16, replace=False)
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)


# 2-2
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, X_inputs)


# 2-3
nce_W = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                          stddev=1.0 / math.sqrt(embedding_size))) 
nce_b = tf.Variable(tf.zeros([vocabulary_size]))
nce_loss = tf.nn.nce_loss(weights=nce_W,
                          biases=nce_b,
                          labels=Y_inputs,
                          inputs=embed,
                          num_sampled=num_sampled,
                          num_classes=vocabulary_size)
loss = tf.reduce_mean(nce_loss)

# 2-4
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

# 2-5
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm # embedding vector normalize
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True) # cosine
