#!/bin/bash

#SBATCH --job-name=ucip # Job name
#SBATCH --gres=gpu:1             # how many gpus would you like to use (here I use 1)
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=16G                  # Job memory request
#SBATCH --time=100:00:00              # Time limit hrs:min:sec
#SBATCH --partition=ava_m.p          # partition name
#SBATCH --output=logs/job_%j.log   # output log
#SBATCH --array=0-2
#SBATCH --exclude=ava-m5,ava-m6,ava-m0 # node to not use

OUTPUT_PATH="output_${SLURM_ARRAY_TASK_ID}.txt"

BMP="./uci-exps/yacht/run-${SLURM_ARRAY_TASK_ID}/"



~/anaconda3/envs/my_laplace/bin/python ./train_uci.py \
    --train_seed 100 \
    --data_seed 100 \
    --act_func leakyrelu \
    --prec_act_func softplus \
    --epochs 250 \
    --mean_warmup 100 \
    --lr 0.01 \
    --lr_max 0.01 \
    --lr_min 0.0001 \
    --hidden_layers 2 \
    --hidden_size 12 \
    --clip 1000. \
    --dataset yacht \
    --step_size_up 2500 \
    --cycle_mode triangular2 \
    --base_model_path $BMP
