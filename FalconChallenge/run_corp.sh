#!/bin/bash

for i in {1..5}
do
    python main.py \
--config ./config/falcon_held_out_eval_corp.yaml \
--eval_data ./nwb/held_out_eval \
--output_path ./data/corp_held_out_eval_seed_${i}.pkl \
--seed $i
done
