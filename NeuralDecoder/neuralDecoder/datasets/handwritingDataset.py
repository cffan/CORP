import pathlib
import random

import numpy as np
import tensorflow as tf

CHAR_DEF = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z',
            '>', ',', "'", '~', '?']

KALDI_CHAR_DEF = ['>', ',', '?', '~', "'",
                  'a', 'b', 'c', 'd', 'e', 'f', 'g',
                  'h', 'i', 'j', 'k', 'l', 'm', 'n',
                  'o', 'p', 'q', 'r', 's', 't',
                  'u', 'v', 'w', 'x', 'y', 'z',
                  ]

class HandwritingDataset():
    def __init__(self,
                 rawFileDir,
                 nInputFeatures,
                 nClasses,
                 maxSeqElements,
                 bufferSize,
                 syntheticFileDir=None,
                 syntheticMixingRate=0.33,
                 subsetSize=-1,
                 labelDir=None,
                 timeWarpSmoothSD=0.0,
                 timeWarpNoiseSD=0.0,
                 chanIndices=None
                 ):
        self.rawFileDir = rawFileDir
        self.nInputFeatures = nInputFeatures
        self.nClasses = nClasses
        self.maxSeqElements = maxSeqElements
        self.bufferSize = bufferSize
        self.syntheticFileDir = syntheticFileDir
        self.syntheticMixingRate = syntheticMixingRate
        self.subsetSize = subsetSize
        self.labelDir = labelDir

    def build(self, batchSize, isTraining):
        def _loadDataset(fileDir, labelDir):
            data_files = sorted([str(x) for x in pathlib.Path(fileDir).glob("*.tfrecord")])
            label_files = None
            if labelDir is not None:
                label_files = sorted([str(x) for x in pathlib.Path(labelDir).glob("*.tfrecord")])
                assert len(data_files) == len(label_files)
            if isTraining:
                shuffle_idx = np.random.permutation(len(data_files))
                data_files = [data_files[i] for i in shuffle_idx]
                if labelDir is not None:
                    label_files = [label_files[i] for i in shuffle_idx]

            if label_files is not None:
                dataset = tf.data.Dataset.zip((tf.data.TFRecordDataset(data_files),
                                               tf.data.TFRecordDataset(label_files)))
            else:
                dataset = tf.data.TFRecordDataset(data_files)
            return dataset

        rawDataset = _loadDataset(self.rawFileDir, self.labelDir)
        if self.syntheticFileDir and self.syntheticMixingRate > 0:
            syntheticDataset = _loadDataset(self.syntheticFileDir, None)
            dataset = tf.data.experimental.sample_from_datasets(
                [rawDataset.repeat(), syntheticDataset.repeat()],
                weights=[1.0 - self.syntheticMixingRate, self.syntheticMixingRate])
        else:
            dataset = rawDataset

        def parseDatasetFunction(*protos):
            datasetFeatures = {
                "inputFeatures": tf.io.FixedLenSequenceFeature([self.nInputFeatures], tf.float32, allow_missing=True),
                #"classLabelsOneHot": tf.io.FixedLenSequenceFeature([self.nClasses+1], tf.float32, allow_missing=True),
                "newClassSignal": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                "ceMask": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                "seqClassIDs": tf.io.FixedLenFeature((self.maxSeqElements), tf.int64),
                "nTimeSteps": tf.io.FixedLenFeature((), tf.int64),
                "nSeqElements": tf.io.FixedLenFeature((), tf.int64),
                "transcription": tf.io.FixedLenFeature((self.maxSeqElements), tf.int64)
            }
            labelFeatures = {
                "seqClassIDs": tf.io.FixedLenFeature((self.maxSeqElements), tf.int64)
            }

            data = tf.io.parse_single_example(protos[0], datasetFeatures)
            if len(protos) == 2:
                label = tf.io.parse_single_example(protos[1], labelFeatures)
                data['seqClassIDs'] = label['seqClassIDs']
            return data

        dataset = dataset.map(parseDatasetFunction, num_parallel_calls=tf.data.AUTOTUNE)

        if isTraining:
            # Use all elements to adapt normalization layer
            datasetForAdapt = dataset.map(lambda x: x['inputFeatures'] + 0.001,
                num_parallel_calls=tf.data.AUTOTUNE)

            # Take a subset of the data if specified
            if self.subsetSize > 0:
                dataset = dataset.take(self.subsetSize)

            dataset = dataset.shuffle(self.bufferSize)
            if self.syntheticMixingRate == 0:
                dataset = dataset.repeat()
            dataset = dataset.padded_batch(batchSize)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            return dataset, datasetForAdapt
        else:
            dataset = dataset.padded_batch(batchSize)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            return dataset
