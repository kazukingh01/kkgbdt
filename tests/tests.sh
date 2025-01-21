#!/bin/bash
set -eu

python train_lgb_cls.py
python train_lgb_reg.py
python train_xgb_cls.py
python train_xgb_reg.py
python tune_lgb_parameter.py
python tune_xgb_parameter.py

# it's not supported
python train_lgb_multi_task.py
python train_lgb_tree_structure.py

# it's for GPU
# python train_xgb_cls_gpu.py
