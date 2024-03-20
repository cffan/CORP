import argparse
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_tfrecord_metric(path, metric_name):
    event_acc = EventAccumulator(path, size_guidance={"tensors": 0})
    event_acc.Reload()

    metrics = pd.DataFrame(
        [
            (tv.step, tf.make_ndarray(tv.tensor_proto).item())
            for tv in event_acc.Tensors(metric_name)
        ],
        columns=["step", metric_name],
    )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str)
    args = parser.parse_args()

    exp_dirs = glob(args.exp_dir)
    cers = []
    for exp_dir in exp_dirs:
        print(exp_dir)
        metrics = extract_tfrecord_metric(exp_dir, "val/seqErrorRate")
        min_idx = metrics["val/seqErrorRate"].idxmin()
        cers.append(metrics.iloc[min_idx]["val/seqErrorRate"])
    print(f"CER: {np.mean(cers):.3f} ± {np.std(cers):.3f}")
