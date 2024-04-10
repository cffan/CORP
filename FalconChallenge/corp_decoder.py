import os
import re
from pathlib import Path
import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.interface import BCIDecoder

import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils
from neuralDecoder.datasets.handwritingDataset import CHAR_DEF
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder, gaussSmooth

from corp_recalibrator import CORPRecalibrator


class CORPDecoder(BCIDecoder):
    def __init__(self, task_config: FalconConfig, corp_config_path: str):
        self.task_config = task_config
        self.corp_config = OmegaConf.load(corp_config_path)
        self.input_buffer = []
        self.zscore_buffer = []
        self.token_def = CHAR_DEF
        self.mode = self.corp_config.mode

        physical_devices = tf.config.list_physical_devices("GPU")
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # Load pretrained model
        self.decoder = self._load_model(
            self.corp_config["init_model_dir"], self.corp_config["init_model_ckpt_idx"]
        )

        if self.mode != "dev":
            # Load LMs
            self._load_ngram_lm(self.corp_config["ngram_dir"])
            self._load_transformer_lm(self.corp_config["gpt2_model"])

        # Create recalibrator
        if self.mode == "online_recal":
            self.recalibrator = CORPRecalibrator(self.corp_config, self.decoder)

    def _load_model(self, init_model_dir, ckpt_idx):
        # Load model config and set which checkpoint to load
        args = OmegaConf.load(os.path.join(init_model_dir, "args.yaml"))
        args["mode"] = "infer"
        args["loadCheckpointIdx"] = ckpt_idx
        args["outputDir"] = init_model_dir

        # Initialize model
        tf.compat.v1.reset_default_graph()
        nsd = NeuralSequenceDecoder(args)

        # Add input layers if needed
        existing_input_layers = nsd.args["dataset"]["datasetToLayerMap"]
        requested_input_layers = self.corp_config["session_input_layers"]
        self._add_input_layers(nsd, existing_input_layers, requested_input_layers)

        return nsd

    def _add_input_layers(self, model, existing_layers, requested_layers):
        layers_to_add = [l for l in requested_layers if l not in existing_layers]
        print(f"Adding input layer {layers_to_add}")
        last_layer = model.inputLayers[-1]
        for l in layers_to_add:
            input_dim = model.args["dataset"]["nInputFeatures"]
            input_layer_size = model.args["model"].get("inputLayerSize", input_dim)
            new_layer = tf.keras.layers.Dense(
                input_layer_size,
                kernel_regularizer=tf.keras.regularizers.L2(
                    model.args["model"]["weightReg"]
                ),
            )
            new_layer.build(input_shape=[input_dim])

            # Copy weights
            from_layer = model.inputLayers[l] if l in existing_layers else last_layer
            for vf, vt in zip(from_layer.variables, new_layer.variables):
                vt.assign(vf)

            model.inputLayers.append(new_layer)

    def _load_ngram_lm(self, ngram_dir):
        if ngram_dir is None:
            return
        if hasattr(self, "ngram_decoder"):
            return

        print(f"Loading ngram LM from {ngram_dir}")
        self.ngram_decoder = lmDecoderUtils.build_lm_decoder(
            ngram_dir,
            acoustic_scale=self.corp_config.acoustic_scale,
            nbest=self.corp_config.nbest,
        )

        self.ngram_rescore = os.path.exists(os.path.join(ngram_dir, "G_no_prune.fst"))

    def _load_transformer_lm(self, model):
        if model is None:
            return
        if hasattr(self, "gpt2_decoder"):
            return

        print(f"Loading gpt2 {model}")
        with tf.device("/cpu:0"):
            self.gpt2_decoder, self.gpt2_tokenizer = lmDecoderUtils.build_gpt2(
                model, cacheDir=self.corp_config.lm_cache_dir
            )

    def _predict_trial(self, feats):
        inputs = feats[None, ...]  # [1, T, C]
        inputs = gaussSmooth(inputs)
        logits = self.decoder.model(self.decoder.inputLayers[self.eval_day_idx](inputs))

        if not hasattr(self, "ngram_decoder"):
            rnn_decoded, _ = tf.nn.ctc_greedy_decoder(
                tf.transpose(logits, [1, 0, 2]), [logits.shape[1]], merge_repeated=True
            )
            rnn_decoded = "".join(
                [
                    self.token_def[c]
                    for c in tf.sparse.to_dense(rnn_decoded[0])[0].numpy()
                ]
            )
            return rnn_decoded, 1.0
        else:
            out = {
                "logits": logits.numpy(),
                "logitLengths": [logits.shape[1]],
            }
            nbests = lmDecoderUtils.nbest_with_lm_decoder(
                self.ngram_decoder,
                out,
                rescore=self.ngram_rescore,
                blankPenalty=np.log(self.corp_config.blank_penalty),
            )
            gpt2_decoded, confidence = lmDecoderUtils.gpt2_lm_rescore(
                self.gpt2_decoder,
                self.gpt2_tokenizer,
                nbests,
                self.corp_config.gpt2_acoustic_scale,
                0,
                self.corp_config.gpt2_alpha,
            )

            confidence = confidence[0]
            gpt2_decoded = gpt2_decoded[0]
            gpt2_decoded = gpt2_decoded.replace(" ", ">")
            gpt2_decoded = gpt2_decoded.replace(".", "~")

            return gpt2_decoded, confidence

    def reset(self, dataset: Path = ""):
        # TODO: dataset path format?
        sess_name = re.search(r"\d{4}\.\d{2}\.\d{2}", dataset.absolute().as_posix()).group()
        sess_idx = self.corp_config.sessions.index(sess_name)
        self.eval_day_idx = self.corp_config.session_input_layers[sess_idx]
        if self.mode == "online_recal":
            self.recalibrator.reset(self.eval_day_idx)
        self.zscore_buffer = []

    def predict(self, neural_feats: np.ndarray) -> np.ndarray:
        # Buffer data
        self.input_buffer.append(neural_feats)
        return None

    def on_trial_end(self):
        # Normalize buffered data
        raw_feats = np.stack(self.input_buffer, axis=0)  # [T, C]
        self.zscore_buffer.extend(self.input_buffer.copy())
        self.zscore_buffer = self.zscore_buffer[-self.corp_config.zscore_window :]
        all_feats = np.stack(self.zscore_buffer, axis=0)
        mean = np.mean(all_feats, axis=0, keepdims=True)
        std = np.std(all_feats, axis=0, keepdims=True)
        norm_feats = (raw_feats - mean) / (std + 1e-8)

        # Predict output
        if self.mode != "dev":
            decoded, confidence = self._predict_trial(norm_feats)
        else:
            decoded = self.gt_transcription
            confidence = 1.0

        # Recalibrate
        if self.mode == "online_recal":
            self.recalibrator.recalibrate(norm_feats, decoded, confidence)

        # Clear states and decoder
        self.input_buffer = []

        # Return output
        return decoded
    
    def observe(self, neural_feats: np.ndarray):
        pass
