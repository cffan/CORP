from glob import glob
import re
from scipy.io import loadmat
from pathlib import Path

from falcon_challenge.config import FalconConfig
from edit_distance import SequenceMatcher

from corp_decoder import CORPDecoder

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
        s = s.replace('>',' ')
        s = s.replace('#','')
        s = re.sub(r"([~,!?])", r" \1", s)
        return s

    total_edit_dist = 0
    total_dist = 0
    for decoded, gt in zip(decoded_transcriptions, gt_transcriptions):
        decoded = _convert(decoded).split(' ')
        gt = _convert(gt).split(' ')
        matcher = SequenceMatcher(a=decoded, b=gt)
        total_edit_dist += matcher.distance()
        total_dist += len(gt)
    return total_edit_dist / total_dist

if __name__ == "__main__":
    task_config = FalconConfig()
    corp_config_path = '/oak/stanford/groups/henderj/stfan/code/CORP/falcon_corp/config/CORP_val.yaml'
    decoder = CORPDecoder(task_config, corp_config_path)
    
    decoded_transcriptions = []
    gt_transcriptions = []
    for sess_data_path in sorted(glob('/oak/stanford/groups/henderj/stfan/data/stability_benchmark/online_recalibration_data/val/*.mat')):
        # Load a new session
        session_data = loadmat(sess_data_path)
        tx_feats = [s for s in session_data['tx_feats'][0]]
        transcriptions = [str(s[0]) for s in session_data['sentences'][0]]
        blocks = [b[0][0] for b in session_data['blocks'][0]]
        
        n_trials = len(tx_feats)

        for trial_id in range(n_trials):
            # print(trial_id, tx_feats[trial_id].shape, transcriptions[trial_id], blocks[trial_id])
            decoder.reset(Path(sess_data_path))
            n_bins = tx_feats[trial_id].shape[0]
            for i in range(n_bins):
                decoder.predict(tx_feats[trial_id][i])

            if decoder.mode == 'dev':
                decoder.gt_transcription = transcriptions[trial_id]
            decoded = decoder.on_trial_end()

            cer = eval_cer([decoded], [transcriptions[trial_id]])
            wer = eval_wer([decoded], [transcriptions[trial_id]])
            print(f'Trial {trial_id} CER {cer:.2f} WER {wer:.2f} \n\tREF: {transcriptions[trial_id]}\n\tHYP: {decoded}')

            decoded_transcriptions.append(decoded)
            gt_transcriptions.append(transcriptions[trial_id])

        # print(f'Average edit distance: {eval_cer(decoded_transcriptions, gt_transcriptions)}')

    # Evaluate
    cer = eval_cer(decoded_transcriptions, gt_transcriptions)
    wer = eval_wer(decoded_transcriptions, gt_transcriptions)
    print(f'Average CER {cer:.2f} WER {wer:.2f}')
    

    # Format eval data into tfrecords