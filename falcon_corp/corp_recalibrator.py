
from glob import glob
import os
from functools import partial
import numpy as np

import tensorflow as tf
from omegaconf import DictConfig

from neuralDecoder.datasets.handwritingDataset import CHAR_DEF
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder, gaussSmooth

LABEL_MAX_LEN = 500

def data_generator(data_buffer):
    for k, data in data_buffer.items():
        for i, d in enumerate(data):
            if 'newClassSignal' in d:
                del d['newClassSignal']
            if 'ceMask' in d:
                del d['ceMask']
            d['layerIdx'] = k

            yield d

class CORPRecalibrator:
    def __init__(self, config: DictConfig, model: NeuralSequenceDecoder):
        self.config = config
        self.nsd = model
        self.prev_data_buffer = {}
        self.curr_data_buffer = []
        self.token_def = CHAR_DEF

        self._load_prev_data()
        self.curr_day_idx = self.config.start_session_idx - 1

    def _load_prev_data(self):
        # Load previous sessions' data and pseudo labels
        for i, sess in enumerate(self.config.sessions[:self.config.start_session_idx]):
            self.prev_data_buffer[self.config.session_input_layers[i]] = \
                CORPRecalibrator._load_data(self.config.seed_model_data_dir, 
                                            sess, 
                                            self.config.feat_dim, 
                                            normalize=False)

    def _init_optimizer(self):
        self.nsd.args['learnRateStart'] = self.config.learning_rate
        self.nsd.args['learnRateEnd'] = self.config.learning_rate
        self.nsd._buildOptimizer()

    @staticmethod
    def _load_data(data_dir, session, feat_dim, normalize=True):
        tfrecords = []
        tf_path = os.path.join(data_dir, session)
        if os.path.exists(os.path.join(data_dir, session, 'train')):
            tf_path = os.path.join(tf_path, 'train')
        tfrecords.extend(glob(os.path.join(tf_path, '*.tfrecord')))
        print(f'Loading following tfrecords')
        print(tfrecords)
        dataset = None
        data_buffer = []
        if len(tfrecords) > 0:
            dataset = tf.data.TFRecordDataset(filenames=tfrecords)
            dataset = dataset.map(partial(CORPRecalibrator._parse_proto, feat_dim), num_parallel_calls=4)
            for d in dataset:
                d['confidences'] = 1.0
                data_buffer.append(d)
        print(f'Loaded {len(data_buffer)} sentences')

        if normalize:
            features = [d['inputFeatures'] for d in data_buffer]
            features = np.concatenate(features, axis=0)
            mean = np.mean(features, axis=0, keepdims=True)
            std = np.std(features, axis=0, keepdims=True)
            for d in data_buffer:
                d['inputFeatures'] = (d['inputFeatures'] - mean) / (std + 1e-8)

        return data_buffer

    @staticmethod
    def _parse_proto(feat_dim, proto):
        datasetFeatures = {
            "inputFeatures": tf.io.FixedLenSequenceFeature([feat_dim], tf.float32, allow_missing=True),
            "newClassSignal": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "ceMask": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "seqClassIDs": tf.io.FixedLenFeature((LABEL_MAX_LEN,), tf.int64),
            "nTimeSteps": tf.io.FixedLenFeature((), tf.int64),
            "nSeqElements": tf.io.FixedLenFeature((), tf.int64),
            "transcription": tf.io.FixedLenFeature((LABEL_MAX_LEN,), tf.int64)
        }

        data = tf.io.parse_single_example(proto, datasetFeatures)
        return data
    
    def _insert_pseudo_labeled_trial(self, feats, pseudo_label, confidence):
        seq_class_ids = np.zeros(LABEL_MAX_LEN, dtype=np.int64)
        transcription = np.zeros(LABEL_MAX_LEN, dtype=np.int64)
        for i, c in enumerate(pseudo_label):
            seq_class_ids[i] = self.token_def.index(c) + 1
            transcription[i] = ord(c)
            
        trial_data = {
            'inputFeatures': tf.constant(feats, dtype=np.float32),
            'seqClassIDs': tf.constant(seq_class_ids),
            'nTimeSteps': tf.constant(feats.shape[0], dtype=np.int64),
            'nSeqElements': tf.constant(len(pseudo_label), dtype=np.int64),
            'transcription': tf.constant(transcription),
            'confidences': tf.constant(confidence, dtype=np.float32)
        }
        self.curr_data_buffer.append(trial_data)

    def _train(self):
        # Compose datasets
        print(f'prev_data_buffer: {sum([len(d) for k, d in self.prev_data_buffer.items()])}')
        print(f'curr_data_buffer: {len(self.curr_data_buffer)}')
        prev_generator = data_generator(self.prev_data_buffer)
        curr_generator = data_generator({
            self.config.session_input_layers[self.curr_day_idx]: self.curr_data_buffer})
        output_signature = {
            'layerIdx': tf.TensorSpec(shape=(), dtype=tf.int32),
            'inputFeatures': tf.TensorSpec(shape=(None, self.config.feat_dim), dtype=tf.float32),
            'seqClassIDs': tf.TensorSpec(shape=(LABEL_MAX_LEN), dtype=tf.int64),
            'nTimeSteps': tf.TensorSpec(shape=(), dtype=tf.int64),
            'nSeqElements': tf.TensorSpec(shape=(), dtype=tf.int64),
            'transcription': tf.TensorSpec(shape=(LABEL_MAX_LEN,), dtype=tf.int64),
            'confidences': tf.TensorSpec(shape=(), dtype=tf.float32)
        }
        prev_dataset = tf.data.Dataset.from_generator(lambda: prev_generator,
                                                      output_signature=output_signature)
        curr_dataset = tf.data.Dataset.from_generator(lambda: curr_generator,
                                                      output_signature=output_signature)
        prev_dataset = prev_dataset.repeat().shuffle(buffer_size=10240)
        curr_dataset = curr_dataset.cache().repeat().shuffle(buffer_size=1024)
        dataset = tf.data.experimental.sample_from_datasets(
            [prev_dataset, curr_dataset],
            weights=[1.0 - self.config.new_data_percent, self.config.new_data_percent]
        )
        dataset = dataset.padded_batch(self.config.batch_size)

        # Train model
        steps = 0
        losses = []
        for data in dataset:
            if steps > self.config.max_train_steps:
                break

            try:
                ctc_loss, reg_loss, total_loss, grad_norm = self._train_step(data['inputFeatures'],
                                                                 data['layerIdx'],
                                                                 data['seqClassIDs'],
                                                                 data['nTimeSteps'],
                                                                 data['confidences'],
                                                                 self.curr_day_idx,
                                                                 self.config.white_noise_sd,
                                                                 self.config.constant_offset_sd,
                                                                 self.config.random_walk_sd,
                                                                 self.config.random_walk_axis,
                                                                 self.nsd.args['smoothKernelSD'],
                )

                print(
                    f'Step: {steps}, ' +
                    f'CTC loss: {ctc_loss.numpy():.4f}, ' +
                    f'Reg loss: {reg_loss.numpy():.4f}, ' +
                    f'Total loss: {total_loss.numpy():.4f}, ' +
                    f'LR: {self.nsd.optimizer._decayed_lr(tf.float32).numpy():.4f}',
                    f'Grad norm: {grad_norm.numpy():.4f}')
                losses.append(total_loss.numpy())
                steps += 1

                if np.mean(losses[-10:]) < self.config.loss_threshold and steps > self.config.min_train_steps:
                    break
            except tf.errors.InvalidArgumentError as e:
                print(e)

        return np.mean(losses), steps
    
    def _train_step(self,
                   inputs,
                   layerIdx,
                   labels,
                   time_steps,
                   confidences,
                   sess_idx,
                   white_noise_sd,
                   constant_offset_sd,
                   random_walk_sd,
                   random_walk_axis,
                   max_seq_len=LABEL_MAX_LEN,
                   grad_clip_value=10.0):
        input_shape = tf.shape(inputs)
        B = input_shape[0]
        C = input_shape[2]

        # Add noise
        inputs += tf.random.normal(shape=input_shape,
                                   mean=0, stddev=white_noise_sd)
        inputs += tf.random.normal([B, 1, C], mean=0,
                                   stddev=constant_offset_sd)
        inputs += tf.math.cumsum(
            tf.random.normal(shape=input_shape, mean=0, stddev=random_walk_sd),
            axis=random_walk_axis)
        inputs = gaussSmooth(inputs)

        # Compute loss
        with tf.GradientTape() as tape:
            new_inputs = []
            for i in range(B):
                new_inputs.append(self.nsd.inputLayers[layerIdx[i]](inputs[i:i+1]))
            new_inputs = tf.concat(new_inputs, axis=0)
            logits = self.nsd.model(new_inputs, training=True)
            sparse_labels = tf.cast(
                tf.sparse.from_dense(labels), dtype=tf.int32)
            sparse_labels = tf.sparse.SparseTensor(
                indices=sparse_labels.indices,
                values=sparse_labels.values-1,
                dense_shape=[1, max_seq_len])
            time_steps = self.nsd.model.getSubsampledTimeSteps(time_steps)

            ctc_loss = tf.compat.v1.nn.ctc_loss_v2(sparse_labels,
                                                   logits,
                                                   None,
                                                   time_steps,
                                                   logits_time_major=False,
                                                   unique=None,
                                                   blank_index=-1,
                                                   name=None)
            ctc_loss = tf.reduce_mean(ctc_loss * confidences)
            reg_loss = tf.math.add_n(
                self.nsd.model.losses) + tf.math.add_n(self.nsd.inputLayers[self.config.session_input_layers[sess_idx]].losses)
            total_loss = ctc_loss + reg_loss

        # Get trainable variables
        if self.config.freeze_backbone:
            trainables = self.nsd.inputLayers[self.config.session_input_layers[sess_idx]].trainable_variables
        else:
            trainables = self.nsd.model.trainable_variables + \
                self.nsd.inputLayers[self.config.session_input_layers[sess_idx]].trainable_variables

        # Apply gradients
        grads = tape.gradient(total_loss, trainables)
        grads, grad_norm = tf.clip_by_global_norm(grads, grad_clip_value)
        self.nsd.optimizer.apply_gradients(zip(grads, trainables))

        return ctc_loss, reg_loss, total_loss, grad_norm
    
    def reset(self, day_idx):
        # Call before session starts

        if day_idx != self.curr_day_idx:
            print(f'Setting recalibrator to day {day_idx}')
            # Create new input layer by copying weights from last layer
            from_layer = self.nsd.inputLayers[self.config.session_input_layers[day_idx - 1]]
            to_layer = self.nsd.inputLayers[self.config.session_input_layers[day_idx]]
            for vf, vt in zip(from_layer.variables, to_layer.variables):
                vt.assign(vf)

            # Move current data to prev data
            if len(self.curr_data_buffer) > 0:
                self.prev_data_buffer[self.config.session_input_layers[self.curr_day_idx]] = self.curr_data_buffer.copy()
                self.curr_data_buffer = []

            self.curr_day_idx = day_idx

    def recalibrate(self, neural_feats, decoded, confidence):
        self._insert_pseudo_labeled_trial(neural_feats, decoded, confidence)
        self._train()

        # Prepare tfrecords for prev data
        # Feed in a new session's data
        # Test whole pipeline