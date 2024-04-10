import numpy as np
from pynwb import NWBHDF5IO


def load_nwb(fn: str):
    print(f"Loading {fn}")
    with NWBHDF5IO(fn, "r") as io:
        nwbfile = io.read()
        session_date = nwbfile.session_start_time.strftime("%Y.%m.%d")
        trial_info = nwbfile.trials.to_dataframe().reset_index()

        binned_spikes = nwbfile.acquisition["binned_spikes"].data[()]
        time = nwbfile.acquisition["binned_spikes"].timestamps[()]
        eval_mask = nwbfile.acquisition["eval_mask"].data[()].astype(bool)

    return (binned_spikes, time, trial_info, eval_mask, session_date)


def extract_trials(binned_spikes, time, sess_info, block_stats={}):
    all_feats = []
    all_cues = []
    blocks = sess_info["block_num"].unique()
    for block in blocks:
        block_feats = []
        block_cues = []
        for idx, row in sess_info[sess_info["block_num"] == block].iterrows():
            start_time_bin = np.argmin(np.abs(time - row["start_time"]))
            end_time_bin = np.argmin(np.abs(time - row["stop_time"]))

            block = row["block_num"]
            cue = row["cue"]
            feats = binned_spikes[start_time_bin:end_time_bin]

            block_feats.append(feats)
            block_cues.append(cue)

        # Normalize with block means and stds
        feat_mean = np.mean(np.concatenate(block_feats, axis=0), axis=0)
        feat_std = np.std(np.concatenate(block_feats, axis=0), axis=0)
        block_stats[block] = (feat_mean, feat_std)
        block_feats = [(f - feat_mean) / (feat_std + 1e-8) for f in block_feats]

        all_feats.extend(block_feats)
        all_cues.extend(block_cues)

    return all_feats, all_cues, block_stats
