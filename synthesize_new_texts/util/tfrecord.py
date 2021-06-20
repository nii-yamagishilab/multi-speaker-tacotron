# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================

import tensorflow as tf
import numpy as np
from collections import namedtuple
from collections.abc import Iterable


class PreprocessedTargetData(namedtuple("PreprocessedTargetData",
                                        ["id", "spec", "spec_width", "mel", "mel_width", "target_length"])):
    pass


class PredictionResult(
    namedtuple("PredictedMel",
               ["id", "key", "predicted_mel", "ground_truth_mel",
                "alignment", "alignments", "source",
                "text"])):
    pass


def bytes_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def int64_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_tfrecord(example: tf.train.Example, filename: str):
    with tf.python_io.TFRecordWriter(filename) as writer:
        writer.write(example.SerializeToString())


def write_preprocessed_target_data(index: int, key: str, spec: np.ndarray, mel: np.ndarray, filename: str):
    raw_spec = spec.tostring()
    raw_mel = mel.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': int64_feature([index]),
        'key': bytes_feature([key.encode('utf-8')]),
        'spec': bytes_feature([raw_spec]),
        'spec_width': int64_feature([spec.shape[1]]),
        'mel': bytes_feature([raw_mel]),
        'mel_width': int64_feature([mel.shape[1]]),
        'target_length': int64_feature([len(mel)]),
    }))
    write_tfrecord(example, filename)


def write_preprocessed_mel_data(index: int, key: str, mel: np.ndarray, filename: str):
    raw_mel = mel.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': int64_feature([index]),
        'key': bytes_feature([key.encode('utf-8')]),
        'mel': bytes_feature([raw_mel]),
        'mel_width': int64_feature([mel.shape[1]]),
        'target_length': int64_feature([len(mel)]),
    }))
    write_tfrecord(example, filename)


def write_preprocessed_mgc_lf0_data(index: int, key: str, mgc: np.ndarray, lf0: np.ndarray, filename: str):
    raw_mgc = mgc.tostring()
    raw_lf0 = lf0.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': int64_feature([index]),
        'key': bytes_feature([key.encode('utf-8')]),
        'mgc': bytes_feature([raw_mgc]),
        'mgc_width': int64_feature([mgc.shape[1]]),
        'lf0': bytes_feature([raw_lf0]),
        'target_length': int64_feature([len(mgc)]),
    }))
    write_tfrecord(example, filename)


def parse_preprocessed_target_data(proto):
    features = {
        'id': tf.FixedLenFeature((), tf.string),
        'spec': tf.FixedLenFeature((), tf.string),
        'spec_width': tf.FixedLenFeature((), tf.int64),
        'mel': tf.FixedLenFeature((), tf.string),
        'mel_width': tf.FixedLenFeature((), tf.int64),
        'target_length': tf.FixedLenFeature((), tf.int64),
    }
    parsed_features = tf.parse_single_example(proto, features)
    return parsed_features


def decode_preprocessed_target_data(parsed):
    spec_width = parsed['spec_width']
    mel_width = parsed['mel_width']
    target_length = parsed['target_length']
    spec = tf.decode_raw(parsed['spec'], tf.float32)
    mel = tf.decode_raw(parsed['mel'], tf.float32)
    return PreprocessedTargetData(
        id=parsed['id'],
        spec=tf.reshape(spec, shape=tf.stack([target_length, spec_width], axis=0)),
        spec_width=spec_width,
        mel=tf.reshape(mel, shape=tf.stack([target_length, mel_width], axis=0)),
        mel_width=mel_width,
        target_length=target_length,
    )


def read_prediction_result(filename: str):
    record_iterator = tf.python_io.tf_record_iterator(filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        id = example.features.feature['id'].int64_list.value[0]
        key = example.features.feature['key'].bytes_list.value[0].decode("utf-8")

        mel_length = example.features.feature['mel_length'].int64_list.value[0]
        mel_width = example.features.feature['mel_width'].int64_list.value[0]
        mel = example.features.feature['mel'].bytes_list.value[0]
        mel = np.frombuffer(mel, dtype=np.float32).reshape([mel_length, mel_width])

        ground_truth_mel_length = example.features.feature['ground_truth_mel_length'].int64_list.value[0]
        ground_truth_mel = example.features.feature['ground_truth_mel'].bytes_list.value[0]
        ground_truth_mel = np.frombuffer(ground_truth_mel, dtype=np.float32).reshape(
            [ground_truth_mel_length, mel_width])

        source_length = example.features.feature['source_length'].int64_list.value[0]
        source = example.features.feature['source'].bytes_list.value[0]
        source = np.frombuffer(source, dtype=np.int64)

        alignments = [np.frombuffer(a, dtype=np.float32) for a in
                      example.features.feature['alignment'].bytes_list.value]

        # source 1
        alignments[0] = alignments[0].reshape([source_length, -1])
        if len(alignments) > 1:
            # source 2
            alignments[1] = alignments[1].reshape([source_length, -1])
            # decoder self attention head 1
            alignments[2] = alignments[2].reshape([mel_length // 2, -1])
            # decoder self attention head 2
            alignments[3] = alignments[3].reshape([mel_length // 2, -1])
            # encoder self attention head 1
            alignments[4] = alignments[4].reshape([source_length, -1])
            # encoder self attention head 2
            alignments[5] = alignments[5].reshape([source_length, -1])

        text = example.features.feature['text'].bytes_list.value[0].decode("utf-8")

        yield PredictionResult(
            id=id,
            key=key,
            predicted_mel=mel,
            ground_truth_mel=ground_truth_mel,
            alignment=alignments[0],
            alignments=alignments,
            source=source,
            text=text,
        )
