#!/bin/bash

#SBATCH --partition=gpunodes        # Specify the GPU partition
#SBATCH --mem=8G                    # Memory allocation
#SBATCH --nodelist=calypso            # Node to use
#SBATCH --gres=gpu:rtx_6000_ada   # Request one RTX 6000 Ada GPU


# Print GPU info to verify allocation
echo "Selected GPU ID(s): ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}"


srun python configs/STFormer/stformer_base_4.py \
    --work_dir "./train_dir_4"
