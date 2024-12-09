#!/bin/bash

srun --partition=gpunodes --mem=16G --nodelist=calypso --gres=gpu:1 \
    python tools/test_davis.py configs/STFormer/stformer_base_test.py --weights=train_dir/checkpoints/epoch_45.pth --work_dir "work_eval/davis_mask_2_epoch_45"

srun --partition=gpunodes --mem=16G --nodelist=calypso --gres=gpu:1 \
    python tools/test_davis.py configs/STFormer/stformer_base_test_4.py --weights=train_dir_4/checkpoints/epoch_45.pth --work_dir "work_eval/davis_mask_4_epoch_45"

srun --partition=gpunodes --mem=16G --nodelist=calypso --gres=gpu:1 \
    python tools/test_davis.py configs/STFormer/stformer_base_test_6.py --weights=train_dir_6/checkpoints/epoch_49.pth --work_dir "work_eval/davis_mask_6_epoch_49"

srun --partition=gpunodes --mem=16G --nodelist=calypso --gres=gpu:1 \
    python tools/test_davis.py configs/STFormer/stformer_base_test_8.py --weights=train_dir_8/checkpoints/epoch_49.pth --work_dir "work_eval/davis_mask_8_epoch_49"

srun --partition=gpunodes --mem=16G --nodelist=calypso --gres=gpu:1 \
    python tools/test.py configs/STFormer/stformer_base.py --weights=train_dir/checkpoints/epoch_45.pth --work_dir "work_dirs/mask_2_epoch_45"

srun --partition=gpunodes --mem=16G --nodelist=calypso --gres=gpu:1 \
    python tools/test.py configs/STFormer/stformer_base_4.py --weights=train_dir_4/checkpoints/epoch_45.pth --work_dir "work_dirs/mask_4_epoch_45"

