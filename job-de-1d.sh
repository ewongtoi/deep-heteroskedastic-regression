#!/bin/bash

#SBATCH --job-name=ffsine # Job name
#SBATCH --gres=gpu:1             # how many gpus would you like to use (here I use 1)
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=16G                  # Job memory request
#SBATCH --time=48:00:00              # Time limit hrs:min:sec
#SBATCH --partition=ava_m.p          # partition name
#SBATCH --output=logs/job_%j.log   # output log


SEED=$((SLURM_ARRAY_TASK_ID + 100))
BMP="./"
echo $BMP

/home/ewongtoi/anaconda3/envs/my_laplace/bin/python train_de_1d.py \
    --train_seed $SEED\
    --data_seed 100 \
    --act_func leakyrelu \
    --scale_act_func exp \
    --epochs 100  \
    --mean_warmup 50 \
    --lr 0.01 \
    --lr_max 0.01 \
    --lr_min 0.00001 \
    --hidden_layers 2 \
    --hidden_size 128 \
    --batch_size 32 \
    --clip 1. \
    --B_dim 64 \
    --B_sigma 2. \
    --dataset sine \
    --samp_size 32 \
    --step_size_up 5000 \
    --cycle_mode triangular2 \
    --base_model_path $BMP

