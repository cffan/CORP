import argparse
import pickle
import re
from glob import glob
from pathlib import Path

from corp_decoder import CORPDecoder
from edit_distance import SequenceMatcher
from falcon_challenge.config import FalconConfig
from scipy.io import loadmat

from nwb_utils import load_nwb, extract_trials


def eval_cer(decoded_transcriptions, gt_transcriptions):
    total_edit_dist = 0
    total_dist = 0
    for decoded, gt in zip(decoded_transcriptions, gt_transcriptions):
        matcher = SequenceMatcher(a=decoded, b=gt)
        total_edit_dist += matcher.distance()
        total_dist += len(gt)
    return total_edit_dist / total_dist


def eval_wer(decoded_transcriptions, gt_transcriptions):
    def _convert(s):
        s = s.replace(">", " ")
        s = s.replace("#", "")
        s = re.sub(r"([~,!?])", r" \1", s)
        return s

    total_edit_dist = 0
    total_dist = 0
    for decoded, gt in zip(decoded_transcriptions, gt_transcriptions):
        decoded = _convert(decoded).split(" ")
        gt = _convert(gt).split(" ")
        matcher = SequenceMatcher(a=decoded, b=gt)
        total_edit_dist += matcher.distance()
        total_dist += len(gt)
    return total_edit_dist / total_dist

def set_seed(seed):
    import random
    import numpy as np
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def run(args):
    set_seed(args.seed)
    task_config = FalconConfig()
    decoder = CORPDecoder(task_config, args.config)

    decoded_transcriptions = []
    gt_transcriptions = []
    for sess_data_path in sorted(glob(f"{args.eval_data}/*.nwb")):
        # Load a new session
        spikes, trial_time, trial_info, _, session_date = load_nwb(sess_data_path)
        print(f"Session {session_date}, {len(spikes)} trials")

        tx_feats, transcriptions, _ = extract_trials(spikes, trial_time, trial_info)
        n_trials = len(tx_feats)

        for trial_id in range(n_trials):
            # print(trial_id, tx_feats[trial_id].shape, transcriptions[trial_id], blocks[trial_id])
            decoder.reset([Path(sess_data_path)])
            n_bins = tx_feats[trial_id].shape[0]
            for i in range(n_bins):
                decoder.predict(tx_feats[trial_id][i].reshape(1, -1))

            if decoder.mode == "dev":
                decoder.gt_transcription = transcriptions[trial_id]
            decoded = decoder.on_done(None)

            cer = eval_cer([decoded], [transcriptions[trial_id]])
            wer = eval_wer([decoded], [transcriptions[trial_id]])
            print(
                f"Trial {trial_id} CER {cer:.2f} WER {wer:.2f} \n\tREF: {transcriptions[trial_id]}\n\tHYP: {decoded}"
            )

            decoded_transcriptions.append(decoded)
            gt_transcriptions.append(transcriptions[trial_id])

        # print(f'Average edit distance: {eval_cer(decoded_transcriptions, gt_transcriptions)}')

    # Evaluate
    cer = eval_cer(decoded_transcriptions, gt_transcriptions)
    wer = eval_wer(decoded_transcriptions, gt_transcriptions)
    print(f"Average CER {cer:.2f} WER {wer:.2f}")

    with open(args.output_path, "wb") as f:
        pickle.dump({
            'decoded': decoded_transcriptions,
            'gt': gt_transcriptions,
        }, f)

    # Save buffered data
    if args.save_val:
        # Save buffered data
        recalibrator = decoder.recalibrator
        recalibrator.prev_data_buffer[
            recalibrator.config.session_input_layers[recalibrator.curr_day_idx]
        ] = recalibrator.curr_data_buffer.copy()
        with open("./data/buffered_data.pkl", "wb") as f:
            pickle.dump(recalibrator.prev_data_buffer, f)

    # Test on test set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_val", action="store_true")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval_data", type=str)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run(args)
