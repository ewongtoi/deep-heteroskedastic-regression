#!/bin/bash

#SBATCH --job-name=sine # Job name
#SBATCH --gres=gpu:1             # how many gpus would you like to use (here I use 1)
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=16G                  # Job memory request
#SBATCH --time=48:00:00              # Time limit hrs:min:sec
#SBATCH --partition=ava_m.p          # partition name
#SBATCH --output=logs/job_%j.log   # output log
#SBATCH --array=0



SEED=$((SLURM_ARRAY_TASK_ID + 100))


BMP="./synth-exps/run-${SLURM_ARRAY_TASK_ID}/"

echo $BMP

python ./train_1d.py \
    --train_seed $SEED \
    --data_seed 1000 \
    --act_func leakyrelu \
    --prec_act_func softplus \
    --epochs 5000 \
    --mean_warmup 2000 \
    --hidden_layers 2 \
    --hidden_size 18 \
    --clip 1000. \
    --dataset sine \
    --samp_size 64 \
    --step_size_up 5000 \
    --cycle_mode triangular2 \
    --base_model_path $BMP
