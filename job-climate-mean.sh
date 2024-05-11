#!/bin/bash

#SBATCH --job-name=prvml # Job name
#SBATCH --gres=gpu:1             # how many gpus would you like to use (here I use 1)
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=16G                  # Job memory request
#SBATCH --time=48:00:00              # Time limit hrs:min:sec
#SBATCH --partition=ava_m.p          # partition name
#SBATCH --output=logs/job_%j.log   # output log
#SBATCH --exclude=ava-m5,ava-m6,ava-m0 # node to not use
#SBATCH --array=0-5

SEED=$((SLURM_ARRAY_TASK_ID + 100))
OUTPUT_PATH="output_${SLURM_ARRAY_TASK_ID}.txt"
#BMP="./climate-exps/run-${SLURM_ARRAY_TASK_ID}/"
BMP="/extra/ucibdl0/ewongtoi/heteroskedastic_nns/climate_exps/solar-flux-local/run-${SLURM_ARRAY_TASK_ID}/"
echo $BMP

~/anaconda3/envs/my_laplace/bin/python ./train_climate_mean.py \
    --train_seed $SEED \
    --data_seed 100 \
    --act_func leakyrelu \
    --prec_act_func softplus \
    --epochs 25 \
    --mean_warmup 10 \
    --batch_size 256 \
    --lr 0.01 \
    --lr_max 0.01 \
    --lr_min 0.0001 \
    --start_factor 0.0001 \
    --total_iters 5000 \
    --hidden_layers 3 \
    --hidden_size 256 \
    --clip 1000. \
    --dataset climate \
    --diag \
    --step_size_up 5000 \
    --cycle_mode triangular2 \
    --base_model_path $BMP