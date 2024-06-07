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
    seed=1,2,3,4,5 \
    'loadDir="/oak/stanford/groups/henderj/stfan/logs/handwriting_logs/corp_falcon_seed_model_heldin/batchSize=48,batchesPerVal=500,dataset=falcon_seed_sessions,model=gru_stack_handwriting,nBatchesToTrain=20000,seed=1"' \
    outputDir=/oak/stanford/groups/henderj/stfan/logs/handwriting_logs/falcon_held_out_few_shot
