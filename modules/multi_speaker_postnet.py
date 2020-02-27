# ==============================================================================
# Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
# Author: Jeff Lai
# All rights reserved.
# ==============================================================================

import tensorflow as tf
from functools import reduce
from tacotron.modules import Conv1d

class MultiSpeakerPostNet(tf.layers.Layer):

    def __init__(self, out_units, speaker_embed, num_postnet_layers, kernel_size, out_channels, is_training,
                 drop_rate=0.5,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(MultiSpeakerPostNet, self).__init__(name=name, trainable=trainable, **kwargs)

        final_conv_layer = Conv1d(kernel_size, out_channels, activation=None, is_training=is_training,
                                  drop_rate=drop_rate,
                                  name=f"conv1d_{num_postnet_layers}",
                                  dtype=dtype)

        self.convolutions = [Conv1d(kernel_size, out_channels, activation=tf.nn.tanh, is_training=is_training,
                                    drop_rate=drop_rate,
                                    name=f"conv1d_{i}",
                                    dtype=dtype) for i in
                             range(1, num_postnet_layers)] + [final_conv_layer]

        self.projection_layer = tf.layers.Dense(out_units, dtype=dtype)
        self.speaker_projection = tf.layers.Dense(out_channels, activation=tf.nn.softsign, dtype=dtype)
        self.speaker_embed = speaker_embed

    def call(self, inputs, **kwargs):
        speaker_embed = tf.expand_dims(self.speaker_projection(self.speaker_embed), axis=1)

        output = reduce(lambda acc, conv: conv(acc) + speaker_embed, self.convolutions, inputs)
        projected = self.projection_layer(output)
        summed = inputs + projected
        return summed

    def compute_output_shape(self, input_shape):
        return self.projection_layer.compute_output_shape(input_shape)
