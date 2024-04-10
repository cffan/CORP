#!/bin/bash

#to run on your machine, you'll need to change the directories specified in
#NeuralDecoder/neuralDecoder/configs/dataset/willett_nature_baseline.yaml
#and change "outputDir" specified below

python -m neuralDecoder.main --multirun hydra/launcher=gpu_slurm_shenoy \
    dataset=willett_nature_baseline \
    batchSize=48 \
    nBatchesToTrain=20000 \
    batchesPerVal=500 \
    model=gru_stack_handwriting \
    model.bidirectional=False \
    seed=1,2,3,4,5 \
    outputDir=/oak/stanford/groups/henderj/stfan/logs/handwriting_logs/willett_nature_baseline
