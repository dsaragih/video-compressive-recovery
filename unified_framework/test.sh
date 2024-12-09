#!/bin/bash

# srun --partition=gpunodes --mem=8G --gres=gpu:rtx_a6000 \
#     python infer_h5.py \
#     --ckpt c2b_optimal.pth \
#     --gpu 0 \
#     --save_gif \
#     --two_bucket

srun --partition=gpunodes --mem=16G --nodelist=calypso --gres=gpu:1 \
    python test.py \
    --ckpt "model_2x2_000198.pth" \
    --gpu 0 \
    --two_bucket \
    --blocksize 2 \
    --subframes 4 \
    --mask_path "./data/2x2_mask.mat" \
    --savedir "results_3"

srun --partition=gpunodes --mem=16G --nodelist=calypso --gres=gpu:1 \
    python test.py \
    --ckpt "model_4x4_000228.pth" \
    --gpu 0 \
    --two_bucket \
    --blocksize 4 \
    --subframes 16 \
    --mask_path "./data/4x4_mask.mat" \
    --savedir "results_3"

srun --partition=gpunodes --mem=16G --nodelist=calypso --gres=gpu:1 \
    python test.py \
    --ckpt "model_6x6_000219.pth" \
    --gpu 0 \
    --two_bucket \
    --blocksize 6 \
    --subframes 36 \
    --mask_path "./data/6x6_mask.mat" \
    --savedir "results_3"

srun --partition=gpunodes --mem=16G --nodelist=calypso --gres=gpu:1 \
    python test.py \
    --ckpt "model_8x8_000199.pth" \
    --gpu 0 \
    --two_bucket \
    --blocksize 8 \
    --subframes 64 \
    --mask_path "./data/8x8_mask.mat" \
    --savedir "results_3"