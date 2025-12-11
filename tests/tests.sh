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

# nohup python tune_lgb_parameter.py --iter 500 --trial 50 --jobs 18 >> tune_lgb.log &
# nohup python tune_xgb_parameter.py --iter 500 --trial 50 --jobs 18 >> tune_xgb.log &
# nohup python tune_cat_parameter.py --iter 500 --trial 20 --jobs 18 >> tune_cat.log &
