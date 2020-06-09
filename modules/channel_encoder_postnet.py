# ==============================================================================
# Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
# Authors: Jeff Lai and Erica Cooper
# All rights reserved.
# ==============================================================================

import tensorflow as tf
from functools import reduce
from tacotron.modules import Conv1d

class ChannelEncoderPostNet(tf.layers.Layer):

    def __init__(self, out_units, channel_code, num_postnet_layers, kernel_size, out_channels, is_training,
                 drop_rate=0.5,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(ChannelEncoderPostNet, self).__init__(name=name, trainable=trainable, **kwargs)

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
        self.channel_code = channel_code

    def call(self, inputs, **kwargs):
        channel_code = tf.expand_dims(self.speaker_projection(self.channel_code), axis=1)

        output = reduce(lambda acc, conv: conv(acc) + channel_code, self.convolutions, inputs)
        projected = self.projection_layer(output)
        summed = inputs + projected
        return summed

    def compute_output_shape(self, input_shape):
        return self.projection_layer.compute_output_shape(input_shape)
