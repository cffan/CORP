#!/bin/bash

#to run on your machine, you'll need to change the directories specified in
#NeuralDecoder/neuralDecoder/configs/dataset/falcon_seed_sessions.yaml
#and change "outputDir" specified below

python -m neuralDecoder.main \
    --multirun hydra/launcher=gpu_slurm_shenoy \
    dataset=falcon_seed_sessions \
    batchSize=48 \
    nBatchesToTrain=20000 \
    batchesPerVal=500 \
    model=gru_stack_handwriting \
    seed=1,2,3,4,5 \
    outputDir=/oak/stanford/groups/henderj/stfan/logs/handwriting_logs/corp_falcon_seed_model_heldin
