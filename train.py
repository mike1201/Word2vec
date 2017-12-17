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
from data_preparation import *
from Model_graph import *

vocabulary_size = 500000
batch_size = 128
embedding_size = 128
skip_window = 2
num_skips = 4
num_sampled = 64
valid_size = 16

# 3
num_steps = 1000001

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    average_loss, data_index = 0, 0
    for step in range(num_steps):
        
        # make batch, labels, data_index
        batch_inputs, batch_labels, data_index = batch_generate(data, batch_size, num_skips, skip_window, data_index)
        _, loss_val = session.run([optimizer, loss],
                                  feed_dict={X_inputs: batch_inputs, Y_inputs: batch_labels})
        average_loss += loss_val

        # Average loss
        if step % 10000 == 0:
            if step > 0:
                average_loss /= 5
            print('Average loss at step {} : {}'.format(step, average_loss))
            average_loss = 0

            # 1similarity pre 10000.
            # if step % 10000 == 0:
            #    sim = similarity.eval()         # (16, 300)
            #
            #    for i in range(valid_size):
            #        valid_word = ordered_words[valid_examples[i]]
            #
            #       top_k = 8
            #        nearest = sim[i].argsort()[-top_k - 1:-1][::-1]
            #        log_str = ', '.join([ordered_words[k] for k in nearest])
            #        print('Nearest to {}: {}'.format(valid_word, log_str))

    final_embeddings = normalized_embeddings.eval()
