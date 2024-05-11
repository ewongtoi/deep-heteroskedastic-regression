#!/bin/bash

#SBATCH --job-name=curNFE # Job name
#SBATCH --gres=gpu:1             # how many gpus would you like to use (here I use 1)
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=1                   # Run a single task		
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=16G                  # Job memory request
#SBATCH --time=26:00:00              # Time limit hrs:min:sec
#SBATCH --partition=ava_m.p          # partition name
#SBATCH --output=logs/job_%j.log   # output log
#SBATCH --array=0

SEED=$((SLURM_ARRAY_TASK_ID + 100))

BMP="./ft-exps/run-${SLURM_ARRAY_TASK_ID}/"
echo $BMP



~/anaconda3/envs/my_laplace/bin/python ./train_ft.py \
    --train_seed $SEED \
    --data_seed 100 \
    --epochs 50 \
    --lr 0.01 \
    --lr_max 0.01 \
    --lr_min 0.0005 \
    --start_factor 0.0001 \
    --clip 1000. \
    --dataset curve \
    --samp_size 128 \
    --step_size_up 50000 \
    --cycle_mode triangular2 \
    --base_model_path $BMP

