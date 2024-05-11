#!/bin/bash

#SBATCH --job-name=3 # Job name
#SBATCH --gres=gpu:1             # how many gpus would you like to use (here I use 1)
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=16G                  # Job memory request
#SBATCH --time=48:00:00              # Time limit hrs:min:sec
#SBATCH --partition=ava_m.p          # partition name
#SBATCH --output=logs/job_%j.log   # output log
#SBATCH --exclude=ava-m5,ava-m6 # node to not use
#SBATCH --array=0-5


SEED=$((SLURM_ARRAY_TASK_ID + 100))


BMP="/extra/ucibdl0/ewongtoi/heteroskedastic_nns/conformal-supp/synth/local/3/run-${SLURM_ARRAY_TASK_ID}/"
echo $BMP

~/anaconda3/envs/my_laplace/bin/python ./train_1d_mean.py \
    --train_seed $SEED \
    --data_seed 1000 \
    --act_func leakyrelu \
    --prec_act_func softplus \
    --epochs 600000 \
    --mean_warmup 250000 \
    --total_iters 50000 \
    --hidden_layers 3 \
    --hidden_size 128 \
    --clip 1000. \
    --dataset 3 \
    --samp_size 64 \
    --step_size_up 1000 \
    --cycle_mode triangular2 \
    --diag \
    --base_model_path $BMP
