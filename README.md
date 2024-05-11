
# Repository for "Understanding Pathologies of Deep Heteroskedastic Regression

## Repo Structure

### Experiments
ft-exps:    directory to store results from running job-parallel-ft.sh
synth-exps: directory to store results from running job-1d.sh
uci-exps:   directory to store results from running job-uci-array.sh

### Jobs
job-parallel-nfe.sh:    slurm file to solve NFE
job-1d.sh: slurm file to launch job that fits neural networks to 1d data
job-uci.sh:      slurm file to launch job to fit neural networks to uci regression datasets



train_ft.py:    solves FT
train_1d.py: fits a neural network to 1d data
train_uci.py:   fits a neural network to uci data
train_climate.py: fits a neural network to 1d data

heteroskedastic_nns:
    load_args.py:      argument loader for experiments
    parallel_ft.py:    implementation of discretized FT
    parallel_model.py: implementation of neural networks
    datasets:          generates the synthetic datasets
    UCI:               holds data for the uci regression datasets/loading them
    utils:
        utils.py:      has useful functions for experiments

Data for ClimSim can be found at: https://leap-stc.github.io/ClimSim/dataset.html
