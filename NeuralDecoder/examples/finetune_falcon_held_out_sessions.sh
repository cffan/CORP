#!/bin/bash

#to run on your machine, you'll need to change the directories specified in
#NeuralDecoder/neuralDecoder/configs/dataset/falcon_held_out_session{}.yaml
#and change "outputDir" specified below

python -m neuralDecoder.main \
    --multirun hydra/launcher=gpu_slurm_shenoy \
    dataset=falcon_held_out_session1,falcon_held_out_session2,falcon_held_out_session3,falcon_held_out_session4,falcon_held_out_session5 \
    batchSize=48 \
    nBatchesToTrain=500 \
    batchesPerSave=50 \
    batchesPerVal=50 \
    model=gru_stack_handwriting \
    'loadDir="/oak/stanford/groups/henderj/stfan/logs/handwriting_logs/corp_falcon_seed_model_minival/batchSize=48,batchesPerVal=500,dataset=corp_falcon_model,model=gru_stack_handwriting,nBatchesToTrain=20000"' \
    outputDir=/oak/stanford/groups/henderj/stfan/logs/handwriting_logs/finetune_falcon_held_out_sessions
