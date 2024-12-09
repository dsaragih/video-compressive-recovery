#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
sbatch dist_train.sh
