#!/bin/bash
set -eu

# training
python train_cat_cls.py
python train_cat_rank.py
python train_cat_reg.py
python train_lgb_cls.py
python train_lgb_multi_task.py
python train_lgb_rank.py
python train_lgb_reg.py
python train_xgb_cls.py
python train_xgb_rank.py
python train_xgb_reg.py

# tuning
python tune_cat_parameter.py --dataset 2 --iter 10 --trial 10 --jobs 4
python tune_lgb_parameter.py --dataset 2 --iter 10 --trial 10 --jobs 4
python tune_xgb_parameter.py --dataset 2 --iter 10 --trial 10 --jobs 4

# nohup python tune_lgb_parameter.py --iter 1000 --trial 40 --jobs 18 > nohup_tune_lgb.log &
# nohup python tune_xgb_parameter.py --iter 1000 --trial 40 --jobs 18 > nohup_tune_xgb.log &
# nohup python tune_cat_parameter.py --iter 1000 --trial 20 --jobs 18 > nohup_tune_cat.log &

# nohup python exp_datasets.py --iter 10000 --entity xxxxxxx-personal --project exp-gbdt-datasets --jobs 56 --nseed 5           > nohup_exp_datasets_init.log &
# nohup python exp_datasets.py --iter 10000 --entity xxxxxxx-personal --project exp-gbdt-datasets --jobs 56 --nseed 10 --isbest > nohup_exp_datasets_best.log &

exit 0

python train_dataset.py \
    --mode lgb --dataset SDSS17 --iter 1000 --trial 40 --jobs 18 --random_seed 1 --nfold 3 --ncv 8 \
    --learning_rate 0.02 --num_leaves 256 --max_depth -1 --min_child_samples 5 \
    --colsample_bytree 1 --colsample_bylevel 1 --max_bin 128 --reg_alpha 0 
