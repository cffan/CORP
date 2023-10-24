import os

import hydra
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from online_recalibrator.online_recalibration_simulator import OnlineRecalibrationSimulator

@hydra.main(config_path='configs', config_name='online_recalibration_simulator_config')
def app(config):

    #set the visible device to the gpu specified in 'args' (otherwise tensorflow will steal all the GPUs)
    if 'Slurm' in HydraConfig.get().launcher._target_:
        # TF train saver doesn't support file name with '[' or ']'. So we'll use relative path here.
        config.output_dir = './'
    print(f'Output dir {config.output_dir}')
    os.makedirs(config.output_dir, exist_ok=True)

    ors = OnlineRecalibrationSimulator(config)
    cer = ors.recalibrate()
    return cer

if __name__ == "__main__":
    app()
