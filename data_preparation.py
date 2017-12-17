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



# 1
# 1-1
def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename



# text --) list
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        names = f.namelist()
        contents = f.read(names[0])
        text = tf.compat.as_str(contents)
        return text.split()


vocabulary = read_data("text8.zip")
print('Data size', len(vocabulary))



# 1-2
# 1-2-1
def build_dataset(words, n_words):
    unique = collections.Counter(words)
    orders = unique.most_common(n_words - 1)
    count = [['UNK', -1]]
    count.extend(orders)
    dictionary = {word: i for i, (word, _) in enumerate(count)}
    # dictionary = { (UNK, 0) (the, 1) (of, 2) (and, 3) (one, 4) (in, 5) (a, 6) (to, 7) }


    data = []
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            count[0][1] += 1
        data.append(index)

    return data, count, list(dictionary.keys())

# 1-2-2
vocabulary_size = 500000
data, count, ordered_words = build_dataset(vocabulary, vocabulary_size)



# 1-3
# 1-3-1
def batch_generate(data, batch_size, num_skips, skip_window, data_index):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window


    # 1-3-1-1
    span = 2 * skip_window + 1
    assert span > num_skips
    buffer = collections.deque(data[data_index:data_index + span], maxlen=span)
    data_index = (data_index + span) % len(data)


    # 1-3-1-2
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    for i in range(batch_size // num_skips):

        # make batch(=X)
        start = i * num_skips  # 0*2=0
        batch[start:start + num_skips] = buffer[skip_window]

        # make labels(=Y)
        targets = list(range(span))
        targets.pop(skip_window)
        np.random.shuffle(targets)
        for j in range(num_skips):
            labels[start + j, 0] = buffer[targets[j]]


        buffer.append(data[data_index])  # [data[0], data[1], data[2]] --) [ data[1], data[2], data[3] ]
        data_index = (data_index + 1) % len(data)  # data_index = 3+1 = 4

    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels, data_index
