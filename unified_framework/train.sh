#!/bin/bash

#SBATCH --partition=gpunodes      # Specify the partition for GPU nodes
#SBATCH --mem=16G                  # Memory allocation
#SBATCH --nodelist=calypso          # Run on the "calypso" node
#SBATCH --gres=gpu:1           # Request one GPU
#SBATCH --time=16:00:00

# Print selected GPU ID
echo "Selected GPUs: ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}"

# srun python train.py \
#         --expt davis \
#         --batch 64 \
#         --gpu 0 \
#         --blocksize 4 \
#         --subframes 16 \
#         --two_bucket \
#         --mask 4x4 \
#         --ckpt models/model_4x4_000129.pth

srun python train.py \
        --expt davis \
        --batch 64 \
        --gpu 0 \
        --blocksize 8 \
        --subframes 64 \
        --two_bucket \
        --mask cc \
        --save_root "models_2"