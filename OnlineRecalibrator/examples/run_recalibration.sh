#!/bin/bash

# To run on your machine, you'll need to change the following arguments:
# output_dir: This is where the recalibration model and results will be saved
# init_model_dir: This is where the seed model is saved
# init_model_ckpt_idx: This is the checkpoint index of the seed model
# ngram_dir: This is where the ngram LM model is saved
# lm_cache_dir: This is cache folder for GPT2 (huggingface will download to this folder)
# data_dir: This is where the tfrecords for the recalibration data is saved
# seed_model_data_dir: This is where the tfrecords for the seed model data is saved

python -m online_recalibrator.online_recalibration_simulator_main \
    output_dir=/oak/stanford/groups/henderj/stfan/logs/handwriting_logs/corp_recal_release \
    init_model_dir=/oak/stanford/groups/henderj/stfan/logs/handwriting_logs/corp_seed_model_release \
    init_model_ckpt_idx=12500 \
    ngram_dir=/scratch/users/stfan/lm_models/handwriting \
    lm_cache_dir=/scratch/users/stfan/huggingface/ \
    data_dir=/oak/stanford/groups/henderj/stfan/data/CORP_data_release/online_evaluation_data/recalibration/tfrecords \
    seed_model_data_dir=/oak/stanford/groups/henderj/stfan/data/CORP_data_release/seed_model_training_data/tfrecords