# ==============================================================================
# Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper (ecooper@nii.ac.jp)
# All rights reserved.
# Load embeddings from a file on disk.
# ==============================================================================

import tensorflow as tf

class ExternalEmbedding(tf.layers.Layer):
    def __init__(self, fname, num_symbols, embedding_dim, index_offset=0,
                 trainable=False, name=None, **kwargs):
        super(ExternalEmbedding, self).__init__(name=name, trainable=trainable, **kwargs)
        self._fname = fname
        self._num_symbols = num_symbols
        self._embedding_dim = embedding_dim
        self.index_offset = tf.convert_to_tensor(index_offset, dtype=tf.int64)

    def build(self, _):
        self._embedding = self.load_embedding_from_file(self._fname)

    def call(self, inputs, **kwargs):
        with tf.control_dependencies([tf.assert_greater_equal(inputs, self.index_offset),
                                      tf.assert_less(inputs, self.index_offset + self._num_symbols)]):
            return tf.nn.embedding_lookup(self._embedding, inputs - self.index_offset)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self._embedding_dim])
                              
    def load_embedding_from_file(self, fname):
        spk_embs = {}
        min = 9999999999  ## please do not have a speaker ID larger than this.
        max = -9999999999 ## we do not assume that embeddings are listed in order.
        vecsize = 0

        f = open(fname, 'r')
        for line in f:
            parts = line.strip().split('  ')
            spkr = int(parts[0][1:])  ## assuming vctk 0.91 speaker ID format for now.
            xvec = [float(x) for x in parts[1].strip('[]').strip().split(' ')]
            if vecsize == 0:
                vecsize = len(xvec)
            if spkr < min:
                min = spkr
            if spkr > max:
                max = spkr
            spk_embs[spkr] = xvec
            
        xv_table = []
        for i in range(min, max+1):
            if i in spk_embs.keys():
                xv_table.append(spk_embs[i])
            else:
                empty = [0 for x in range(0, vecsize)]  ## placeholder for skipped speaker IDs
                xv_table.append(empty)

        return tf.constant(xv_table)
