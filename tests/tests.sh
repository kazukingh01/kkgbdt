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

# nohup python exp_datasets.py --iter 5000 --entity xxxxxxx-personal --project exp-gbdt-datasets --jobs 56 --nseed 5           > nohup_exp_datasets_init.log &
# nohup python exp_datasets.py --iter 5000 --entity xxxxxxx-personal --project exp-gbdt-datasets --jobs 56 --nseed 10 --isbest > nohup_exp_datasets_best.log &