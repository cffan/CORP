from scipy.io import loadmat

from falcon_challenge.config import FalconConfig

# from corp_decoder import CORPDecoder

if __name__ == "__main__":
    task_config = FalconConfig()
    # corp_config_path = 'config/CORP_val.yaml'
    # decoder = CORPDecoder(task_config, corp_config_path)
    
    # Load a new session
    session_data = loadmat('/Users/cfan/Downloads/stability_benchmark/online_recalibration_data/t5.2022.09.29.mat')
    tx_feats = [s for s in session_data['tx_feats'][0]]
    transcriptions = [str(s[0]) for s in session_data['sentences'][0]]
    blocks = [b[0][0] for b in session_data['blocks'][0]]
    
    n_trials = len(tx_feats)
    for trial_id in range(n_trials):
        print(trial_id, tx_feats[trial_id].shape, transcriptions[trial_id], blocks[trial_id])
    

    # Run decoder on the session
    # Eval results