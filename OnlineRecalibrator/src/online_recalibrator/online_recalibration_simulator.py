import os
import pickle
import random
from functools import partial
from glob import glob
from pathlib import Path

import numpy as np
import tensorflow as tf
from edit_distance import SequenceMatcher
from omegaconf import OmegaConf
from scipy.ndimage.filters import gaussian_filter1d

import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils
from neuralDecoder.datasets.handwritingDataset import CHAR_DEF
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder


@tf.function(experimental_relax_shapes=True)
def gauss_smooth(inputs, kernelSD=2, padding='SAME'):
    """
    Applies a 1D gaussian smoothing operation with tensorflow to smooth the data along the time axis.

    Args:
        inputs (tensor : B x T x N): A 3d tensor with batch size B, time steps T, and number of features N
        kernelSD (float): standard deviation of the Gaussian smoothing kernel

    Returns:
        smoothedData (tensor : B x T x N): A smoothed 3d tensor with batch size B, time steps T, and number of features N
    """

    #get gaussian smoothing kernel
    inp = np.zeros([100], dtype=np.float32)
    inp[50] = 1
    gaussKernel = gaussian_filter1d(inp, kernelSD)
    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel/np.sum(gaussKernel))

    # Apply depth_wise convolution
    B, T, C = inputs.shape.as_list()
    filters = tf.tile(gaussKernel[None, :, None, None], [1, 1, C, 1])  # [1, W, C, 1]
    inputs = inputs[:, None, :, :]  # [B, 1, T, C]
    smoothedInputs = tf.nn.depthwise_conv2d(inputs, filters, strides=[1, 1, 1, 1], padding=padding)
    smoothedInputs = tf.squeeze(smoothedInputs, 1)

    return smoothedInputs

def load_model(init_model_dir, ckpt_idx, dropout):
    cwd = os.getcwd()
    os.chdir(init_model_dir)

    # Load model config and set which checkpoint to load
    args = OmegaConf.load(os.path.join(init_model_dir, 'args.yaml'))
    args['mode'] = 'infer'
    args['loadCheckpointIdx'] = ckpt_idx
    args['model']['dropout'] = dropout

    # Initialize model
    tf.compat.v1.reset_default_graph()
    nsd = NeuralSequenceDecoder(args)

    os.chdir(cwd)

    return nsd

def add_input_layers(model, existing_layers, requested_layers):
    layers_to_add = [l for l in requested_layers if l not in existing_layers]
    print(f'Adding input layer {layers_to_add}')
    last_layer = model.inputLayers[-1]
    for l in layers_to_add:
        input_dim = model.args['dataset']['nInputFeatures']
        input_layer_size = model.args['model'].get('inputLayerSize', input_dim)
        if 'inputNetwork' in model.args['model']:
            new_layer = tf.keras.Sequential()
            new_layer.add(tf.keras.Input(shape=(None, input_dim)))
            new_layer.add(tf.keras.layers.Dense(input_layer_size,
                                                activation=model.args['model']['inputNetwork']['activation'],
                                                kernel_regularizer=tf.keras.regularizers.L2(model.args['model']['weightReg']))
            )
            new_layer.add(tf.keras.layers.Dropout(model.args['model']['inputNetwork']['dropout']))
        else:
            new_layer = tf.keras.layers.Dense(input_layer_size,
                                              kernel_regularizer=tf.keras.regularizers.L2(model.args['model']['weightReg']))
            new_layer.build(input_shape=[input_dim])

        # Copy weights
        from_layer = model.inputLayers[l] if l in existing_layers else last_layer
        for vf, vt in zip(from_layer.variables, new_layer.variables):
            vt.assign(vf)

        model.inputLayers.append(new_layer)

def parse_proto(feat_dim, proto):
    datasetFeatures = {
        "inputFeatures": tf.io.FixedLenSequenceFeature([feat_dim], tf.float32, allow_missing=True),
        "newClassSignal": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        "ceMask": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        "seqClassIDs": tf.io.FixedLenFeature((500,), tf.int64),
        "nTimeSteps": tf.io.FixedLenFeature((), tf.int64),
        "nSeqElements": tf.io.FixedLenFeature((), tf.int64),
        "transcription": tf.io.FixedLenFeature((500,), tf.int64)
    }

    data = tf.io.parse_single_example(proto, datasetFeatures)
    return data

def replace_label(data, label, token_def, task, confidence):
    data['seqClassIDs'] = np.zeros_like(data['seqClassIDs'])
    data['transcription'] = np.zeros_like(data['transcription'])
    if task == 'handwriting':
        for j, c in enumerate(label):
            data['seqClassIDs'][j] = token_def.index(c) + 1
            data['transcription'][j] = ord(c)
            data['confidences'] = confidence

def randomly_change_label(label, token_def, prob):
    new_label = []
    for c in label:
        if np.random.rand() > prob:
            new_label.append(c)
        else:
            change_type = np.random.choice(['insert', 'delete', 'replace'])
            if change_type == 'insert':
                new_label.append(np.random.choice(token_def))
                new_label.append(c)
            elif change_type == 'replace':
                new_label.append(np.random.choice(token_def))
            elif change_type == 'delete':
                continue
    return ''.join(new_label)

def data_generator(data_buffer):
    for k, data in data_buffer.items():
        for i, d in enumerate(data):
            if 'newClassSignal' in d:
                del d['newClassSignal']
            if 'ceMask' in d:
                del d['ceMask']
            d['layerIdx'] = k

            yield d

def load_data(data_dir, session, feat_dim, normalize=True):
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
        dataset = dataset.map(partial(parse_proto, feat_dim), num_parallel_calls=4)
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

class OnlineRecalibrationSimulator:

    def __init__(self, config):
        self.config = config
        self.sessions = config['sessions']
        self.data_dir = config['data_dir']
        self.seed_model_data_dir = config['seed_model_data_dir']
        self.output_dir = config['output_dir']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.curr_percent = config['new_data_percent']
        self.max_train_steps = config['max_train_steps']
        self.min_train_steps = config['min_train_steps']
        self.recalibration = config['recalibration']
        self.pseudo_label_method = config['pseudo_label_method']
        self.num_norm_sentences = config['num_norm_sentences']
        self.start_session_idx = config['start_session_idx']
        self.loss_threshold = config['loss_threshold']
        self.white_noise_sd = config['white_noise_sd']
        self.constant_offset_sd = config['constant_offset_sd']
        self.random_walk_sd = config['random_walk_sd']
        self.random_walk_axis = config['random_walk_axis']
        self.lm_cache_dir = config['lm_cache_dir']

        seed = config.get('seed', 0)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        self.token_def = CHAR_DEF

        with open(os.path.join(self.output_dir, 'args.yaml'), 'w') as f:
            OmegaConf.save(config=config, f=f)

        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # Load pretrained model
        self.nsd = load_model(config['init_model_dir'],
                              config['init_model_ckpt_idx'],
                              dropout=self.config.get('dropout', 0.4))
        # Add input layers if needed
        existing_input_layers = self.nsd.args['dataset']['datasetToLayerMap']
        requested_input_layers = self.config['session_input_layers']
        add_input_layers(self.nsd, existing_input_layers, requested_input_layers)

        self.checkpoint = tf.train.Checkpoint(net=self.nsd.model,
                                              optimizer=self.nsd.optimizer,
                                              input_layers=self.nsd.inputLayers)

        if Path(self.output_dir, 'checkpoint').exists():
            print('Restoring from checkpoint')
            self.checkpoint.restore(
                tf.train.latest_checkpoint(self.output_dir))
        self.ckpt_manager = tf.train.CheckpointManager(
            self.checkpoint, self.output_dir, max_to_keep=1)

        # Load training state
        output_path = Path(self.output_dir)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        self.state = {
            'current_session': self.start_session_idx,
            'current_sentence': 0,
            'session': [],
            'sentence_idx': [],
            'sentence': [],
            'decoded': [],
            'lm_decoded': [],
            'pseudo_label': [],
            'cer': [],
            'wer': [],
            'lm_cer': [],
            'lm_wer': [],
            'lm_confidence': [],
            'loss': [],
            'train_steps': [],
        }
        self.state_path = Path(self.output_dir, 'state.pkl')
        self.load_state()

        if self.pseudo_label_method == 'ngram':
            self.load_ngram_lm(self.config['ngram_dir'])
        else:
            self.load_ngram_lm(self.config['ngram_dir'])
            self.load_transformer_lm(self.config['gpt2_model'])

    def load_state(self):
        if self.state_path.exists():
            with self.state_path.open('rb') as f:
                self.state = pickle.load(f)

    def save_state(self):
        with self.state_path.open('wb') as f:
            pickle.dump(self.state, f)

    def load_ngram_lm(self, ngram_dir):
        if hasattr(self, 'ngram_decoder'):
            return

        print(f'Loading ngram LM from {ngram_dir}')
        self.ngram_decoder = lmDecoderUtils.build_lm_decoder(
            ngram_dir,
            acoustic_scale=self.config.acoustic_scale,
            nbest=self.config.nbest
        )

        self.ngram_rescore = os.path.exists(os.path.join(ngram_dir, 'G_no_prune.fst'))

    def load_transformer_lm(self, model):
        if hasattr(self, 'gpt2_decoder'):
            return

        if 'gpt' in model:
            print(f'Loading gpt2 {model}')
            with tf.device('/cpu:0'):
                self.gpt2_decoder, self.gpt2_tokenizer = lmDecoderUtils.build_gpt2(
                    model, cacheDir=self.lm_cache_dir
                )
        elif 'opt' in model:
            print(f'Loading opt {model}')
            with tf.device('/device:GPU:0'):
                self.gpt2_decoder, self.gpt2_tokenizer = lmDecoderUtils.build_opt(
                    model, cacheDir=self.lm_cache_dir
                )

    def eval_average_error_rate(self):
        total_edist = 0
        total_len = 0
        if self.config.task == 'handwriting':
            for ref, hyp in zip(self.state['sentence'], self.state['decoded']):
                matcher = SequenceMatcher(a=ref, b=hyp)
                total_edist += matcher.distance()
                total_len += len(ref)
        return total_edist / total_len

    def recalibrate(self):
        sess_idx = self.state['current_session']
        while sess_idx < len(self.sessions):
            print(f'Running sessions {self.sessions[sess_idx]}')

            # Load previous sessions' data and pseudo labels
            prev_data_buffer = {}
            pseudo_label_count = 0
            for i, sess in enumerate(self.sessions[:sess_idx]):
                if i < self.start_session_idx:
                    prev_data_buffer[self.config.session_input_layers[i]] = \
                        load_data(self.seed_model_data_dir, sess, self.config.feat_dim, normalize=False)
                else:
                    sess_data = load_data(self.data_dir, sess, self.config.feat_dim, normalize=True)
                    if self.pseudo_label_method is not None:
                        # Must have enough pseudo labels left for this session
                        assert len(self.state['pseudo_label']) >= (pseudo_label_count + len(sess_data))
                        sess_pseudo_labels = self.state['pseudo_label'][pseudo_label_count:pseudo_label_count+len(sess_data)]
                        sess_pseudo_label_confs = self.state['lm_confidence'][pseudo_label_count:pseudo_label_count+len(sess_data)]
                        assert len(sess_data) == len(sess_pseudo_labels)
                        for d, l, c in zip(sess_data, sess_pseudo_labels, sess_pseudo_label_confs):
                            replace_label(d, l, self.token_def, self.config.task, c)
                        pseudo_label_count += len(sess_data)
                    prev_data_buffer[self.config.session_input_layers[i]] = sess_data

            # Load current session's data
            curr_data_buffer = load_data(
                self.data_dir, self.sessions[sess_idx], self.config.feat_dim, normalize=False)
            print(f'Current sentences: {len(curr_data_buffer)}')

            # Load current session's pseudo labels
            sentence_idx = self.state['current_sentence']
            if sentence_idx > 0 and self.pseudo_label_method is not None:
                # Remaining pseudo lables must equal to the labeled sentences
                assert len(self.state['pseudo_label']) - pseudo_label_count == sentence_idx
                curr_pseudo_labels = self.state['pseudo_label'][pseudo_label_count:]
                curr_pseudo_label_confs = self.state['lm_confidence'][pseudo_label_count:]
                for d, l, c in zip(curr_data_buffer, curr_pseudo_labels, curr_pseudo_label_confs):
                    replace_label(d, l, self.token_def, self.config.task, c)

            # Rolling normalization of current session's data
            input_feature_buffer = []
            for i in range(len(curr_data_buffer)):
                input_feature_buffer.append(
                    curr_data_buffer[i]['inputFeatures'])
                if i == 0:
                    input_features = input_feature_buffer[0]
                else:
                    input_features = np.concatenate(
                        input_feature_buffer[max(0, i - self.num_norm_sentences):i], axis=0)
                mean = np.mean(input_features, axis=0, keepdims=True)
                std = np.std(input_features, axis=0, keepdims=True)
                curr_data_buffer[i]['inputFeatures'] = (
                    curr_data_buffer[i]['inputFeatures'] - mean) / (std + 1e-8)

            # Init optimizer
            self.nsd.args['learnRateStart'] = self.learning_rate
            self.nsd.args['learnRateEnd'] = self.learning_rate
            self.nsd._buildOptimizer()
            self.checkpoint = tf.train.Checkpoint(net=self.nsd.model,
                                                  optimizer=self.nsd.optimizer,
                                                  input_layer=self.nsd.inputLayers)
            self.ckpt_manager = tf.train.CheckpointManager(
                self.checkpoint, self.output_dir, max_to_keep=1)

            # Copy previous session's input layer weights
            if sentence_idx == 0:
                from_layer = self.nsd.inputLayers[self.config.session_input_layers[sess_idx - 1]]
                to_layer = self.nsd.inputLayers[self.config.session_input_layers[sess_idx]]
                for vf, vt in zip(from_layer.variables, to_layer.variables):
                    vt.assign(vf)

            # Start simulation
            while sentence_idx < len(curr_data_buffer):
                curr_sentence_data = curr_data_buffer[sentence_idx]

                # Eval current sentence using previous model
                rnn_decoded, lm_decoded, cer, lm_cer, lm_wer, lm_confidence = \
                    self.eval_single_sentence(curr_sentence_data, self.config.session_input_layers[sess_idx])
                self.state['cer'].append(cer)
                self.state['wer'].append(lm_wer)
                self.state['session'].append(self.sessions[sess_idx])
                self.state['sentence_idx'].append(sentence_idx)
                self.state['sentence'].append(''.join(
                    [chr(c) for c in curr_sentence_data['transcription'].numpy() if c > 0]))
                self.state['decoded'].append(rnn_decoded)
                self.state['lm_decoded'].append(lm_decoded)
                print(f"session: {self.state['current_session']}, trial: {self.state['current_sentence']}, cer: {self.state['cer'][-1]}")
                print(f'rnn decoded: {rnn_decoded}')
                print(f'lm decoded: {lm_decoded}')

                # Pseudo-label
                if self.pseudo_label_method is not None:
                    if self.pseudo_label_method == 'rnn':
                        self.state['pseudo_label'].append(self.state['decoded'][-1])
                    elif self.pseudo_label_method == 'ngram' or self.pseudo_label_method == 'gpt2':
                        self.state['pseudo_label'].append(lm_decoded)
                        self.state['lm_cer'].append(lm_cer)
                        self.state['lm_wer'].append(lm_wer)
                        self.state['lm_confidence'].append(lm_confidence)

                    if len(self.state['pseudo_label'][-1]) == 0:
                        # Empty pseudo label will cause training to crash.
                        # Use '.' or '>' instead.
                        # TODO: fix this hack
                        self.state['pseudo_label'][-1] = '>'

                    replace_label(curr_sentence_data,
                                  self.state['pseudo_label'][-1],
                                  self.token_def,
                                  self.config.task,
                                  lm_confidence if self.config.use_lm_confidence else 1.0)
                    print(f'Pseudo-label: {self.state["pseudo_label"][-1]}')
                    print(f"GT label: {self.state['sentence'][-1]}")

                if self.config.get('label_change_prob', 0.0) > 0:
                    assert self.pseudo_label_method is None  # Only for GT labels
                    new_label = randomly_change_label(self.state['sentence'][-1],
                                                      self.token_def,
                                                      self.config.get('label_change_prob', 0.0))
                    replace_label(curr_sentence_data,
                                  new_label,
                                  self.token_def,
                                  self.config.task,
                                  1.0)
                    print(f'Changed label: {new_label}')
                    print(f"GT label: {self.state['sentence'][-1]}")

                if self.recalibration:
                    # Fine tune model
                    avg_loss, train_steps = self.train(
                        prev_data_buffer,
                        curr_data_buffer[:sentence_idx+1],
                        sess_idx
                    )

                    cer_after = 0
                    lm_cer_after = 0
                    lm_wer_after = 0
                    print(f'CER before: {cer:.4f}, after: {cer_after:.4f}, '
                          f'LM CER before: {lm_cer:.4f}, after: {lm_cer_after:.4f} '
                          f'LM WER before: {lm_wer:.4f}, after: {lm_wer_after:.4f} '
                          f'avg loss: {avg_loss:.4f}',
                          f'train steps: {train_steps}')
                    self.state['loss'].append(avg_loss)
                    self.state['train_steps'].append(train_steps)

                # Save checkpoint and state
                self.ckpt_manager.save()
                sentence_idx += 1
                self.state['current_sentence'] = sentence_idx
                self.save_state()

            if self.config.early_stop_criteria is not None:
                average_error_rate = self.eval_average_error_rate()
                if average_error_rate >= self.config.early_stop_criteria:
                    print(f'Early stop at session {sess_idx}, CER: {average_error_rate}')
                    return average_error_rate

            # Save state
            sess_idx += 1
            sentence_idx = 0
            self.state['current_session'] = sess_idx
            self.state['current_sentence'] = sentence_idx
            self.save_state()

        average_error_rate = self.eval_average_error_rate()
        print(f'Final CER: {average_error_rate}')

        return average_error_rate

    def eval_single_sentence(self, data, layer_idx):
        # Inference
        inputs = data['inputFeatures'][None, ...]
        inputs = gauss_smooth(inputs)
        logits = self.nsd.model(self.nsd.inputLayers[layer_idx](inputs))

        # Eval RNN CER
        sparse_labels = tf.sparse.from_dense(data['seqClassIDs'][None, ...])
        sparse_labels = tf.sparse.SparseTensor(
            indices=sparse_labels.indices,
            values=sparse_labels.values-1,
            dense_shape=[1, self.nsd.args['dataset']['maxSeqElements']])
        time_steps = self.nsd.model.getSubsampledTimeSteps(data['nTimeSteps'][None, ...])
        rnn_decoded, _ = tf.nn.ctc_greedy_decoder(tf.transpose(logits, [1, 0, 2]),
                                                  time_steps,
                                                  merge_repeated=True)
        edit_dist = tf.edit_distance(
            rnn_decoded[0], sparse_labels, normalize=False)
        rnn_cer = tf.cast(tf.reduce_sum(edit_dist), tf.float32) / \
            tf.cast(tf.reduce_sum(data['nSeqElements']), tf.float32)
        rnn_decoded = ' '.join([self.token_def[c] for c in tf.sparse.to_dense(rnn_decoded[0])[0].numpy()])
        if self.config.task == 'handwriting':
            rnn_decoded = rnn_decoded.replace(' ', '')
        print(f'RNN cer: {rnn_cer}')

        lm_decoded = None
        lm_cer = 0.0
        lm_wer = 0.0
        lm_confidence = 1.0
        out = {
            'logits': logits.numpy(),
            'logitLengths': [logits.shape[1]],
            'transcriptions': [data['transcription']],
            'trueSeqs': [data['seqClassIDs'] - 1]
        }
        outputType = self.config.task

        if hasattr(self, 'gpt2_decoder') and hasattr(self, 'ngram_decoder'):
            nbestOutputs = lmDecoderUtils.nbest_with_lm_decoder(self.ngram_decoder,
                                                               out,
                                                               outputType=outputType,
                                                               rescore=self.ngram_rescore,
                                                               blankPenalty=np.log(self.config.blank_penalty))
            gpt2Outputs = lmDecoderUtils.cer_with_gpt2_decoder(self.gpt2_decoder,
                                                               self.gpt2_tokenizer,
                                                               nbestOutputs,
                                                               self.config.gpt2_acoustic_scale,
                                                               out,
                                                               alpha=self.config.gpt2_alpha,
                                                               outputType=outputType)
            print(f"gpt2 cer {gpt2Outputs['cer']}, wer {gpt2Outputs['wer']}, confidence: {gpt2Outputs['confidences']}")
            lm_cer = gpt2Outputs['cer']
            lm_wer = gpt2Outputs['wer']
            lm_confidence = gpt2Outputs['confidences'][0]
            lm_decoded = gpt2Outputs['decoded_transcripts'][0].strip()
            if self.config.task == 'handwriting':
                lm_decoded = lm_decoded.replace(' ', '>')
                lm_decoded = lm_decoded.replace('.', '~')
        elif hasattr(self, 'ngram_decoder'):
            lm_outputs = lmDecoderUtils.cer_with_lm_decoder(self.ngram_decoder,
                                                            out,
                                                            outputType=outputType,
                                                            rescore=self.ngram_rescore,
                                                            blankPenalty=np.log(self.config.blank_penalty))
            print(f"ngram cer {lm_outputs['cer']}, wer {lm_outputs['wer']}")
            lm_cer = lm_outputs['cer']
            lm_wer = lm_outputs['wer']
            lm_confidence = 1.0
            lm_decoded = lm_outputs['decoded_transcripts'][0].strip()

        return rnn_decoded, lm_decoded, rnn_cer, lm_cer, lm_wer, lm_confidence

    def train(self, prev_data_buffer, curr_data_buffer, curr_sess_idx):
        # Compose datasets
        print(f'prev_data_buffer: {sum([len(d) for k, d in prev_data_buffer.items()])}')
        print(f'curr_data_buffer: {len(curr_data_buffer)}')
        prev_generator = data_generator(prev_data_buffer)
        if self.config.num_curr_sentences > 0:
            curr_data_buffer = curr_data_buffer[-self.config.num_curr_sentences:]
        curr_generator = data_generator({
            self.config.session_input_layers[curr_sess_idx]: curr_data_buffer})
        output_signature = {
            'layerIdx': tf.TensorSpec(shape=(), dtype=tf.int32),
            'inputFeatures': tf.TensorSpec(shape=(None, self.config.feat_dim), dtype=tf.float32),
            'seqClassIDs': tf.TensorSpec(shape=(500), dtype=tf.int64),
            'nTimeSteps': tf.TensorSpec(shape=(), dtype=tf.int64),
            'nSeqElements': tf.TensorSpec(shape=(), dtype=tf.int64),
            'transcription': tf.TensorSpec(shape=(500,), dtype=tf.int64),
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
            weights=[1.0 - self.curr_percent, self.curr_percent]
        )
        dataset = dataset.padded_batch(self.batch_size)

        # Train model
        steps = 0
        losses = []
        for data in dataset:
            if steps > self.min_train_steps and steps > self.max_train_steps:
                break

            try:
                ctc_loss, reg_loss, total_loss, grad_norm = self.train_step(data['inputFeatures'],
                                                                 data['layerIdx'],
                                                                 data['seqClassIDs'],
                                                                 data['nTimeSteps'],
                                                                 data['confidences'],
                                                                 curr_sess_idx,
                                                                 self.white_noise_sd,
                                                                 self.constant_offset_sd,
                                                                 self.random_walk_sd,
                                                                 self.random_walk_axis,
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

                if np.mean(losses[-10:]) < self.loss_threshold and steps > self.min_train_steps:
                    break
            except tf.errors.InvalidArgumentError as e:
                print(e)

        return np.mean(losses), steps

    def train_step(self,
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
                   max_seq_len=500,
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
        inputs = gauss_smooth(inputs)

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