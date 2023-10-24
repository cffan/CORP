#!/bin/bash

#to run on your machine, you'll need to change the directories specified in
#NeuralDecoder/neuralDecoder/configs/dataset/corp_seed_model_release.yaml
#and change "outputDir" specified below

python -m neuralDecoder.main \
    dataset=corp_seed_model_release \
    batchSize=48 \
    nBatchesToTrain=20000 \
    batchesPerVal=500 \
    model=gru_stack_handwriting \
    outputDir=/oak/stanford/groups/henderj/stfan/logs/handwriting_logs/corp_seed_model_release
