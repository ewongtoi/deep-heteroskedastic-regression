import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import numpy as np
import dill as pickle
import json
from tqdm import tqdm

import sys
import heteroskedastic_nns.datasets.load_uci
from heteroskedastic_nns.load_args import get_args
from heteroskedastic_nns.parallel_model import ParallelFF
from heteroskedastic_nns.datasets.generate_data import prep_double_sine, prep_polynomial_data, prep_uci_data, prep_synth_data
from heteroskedastic_nns.utils.utils import run_exp, plot_result, plot_sd_res, make_heatmap, plot_parallel_model, gam_rho_to_alpha_beta

def main(args):
    print('main')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    het_noise = not args.homoskedastic_noise
    x_all, y_all, _ = prep_synth_data(args.dataset, device, args.data_seed, args.samp_size, het_noise)

    plt.plot(x_all.cpu().detach(), y_all.cpu().detach())
    plt.savefig(args.base_model_path + 'data.png')

    

    pickle.dump(x_all, open(args.base_model_path + 'x_all.p', 'wb'))
    pickle.dump(y_all, open(args.base_model_path + 'y_all.p', 'wb'))



    if args.diag:    
        symlog = [1.0, 0.99999999999, 0.9999999999, 0.999999999, 0.99999999, 0.9999999, 0.999999, 0.99999, 0.9999, 0.999, 0.99] + [sl * .01 + .1 for sl in range(81)] + [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001, 0.00000000001, 0.0]
        gamma = symlog[1:-1]

        rho = [0.5, 0.6]
    
    if args.beta_nll:
        symlog = [1.0, 1.0, 1.0, 1.0, 1.0]
        gamma = symlog[1:-1]
        rho = [i for i in reversed(symlog[1:-1])]
    
    if args.mle:
        symlog = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        gamma = symlog
        rho = symlog

    if not args.diag and not (args.beta_nll or args.mle):
        symlog = [1.0, 1.0, 0.9999, 0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001, 0.00000000001, 0.0, 0.0]
        gamma = symlog[1:-1]
        rho = [i for i in reversed(symlog[1:-1])]        


    print('ppml')
    print(args.per_param_loss)

    fail_it, parallel_model = run_exp(x_all[::2,:], y_all[::2,:], device, args.train_seed, gamma, rho, 1, 1,
            args.act_func, args.prec_act_func, args.epochs, args.lr, args.momentum,
            args.lr_min, args.lr_max, args.cycle_mode, args.base_model_path + 'first/', 
            pre_trained_path=args.pre_trained_path, per_param_loss=args.per_param_loss,
            hidden_layers=args.hidden_layers, hidden_size=args.hidden_size,
            step_size_up=args.step_size_up, clip=args.clip, mean_warmup=args.mean_warmup, 
            beta_nll=args.beta_nll, var_param=args.var_param, diag=args.diag)
    
    print(fail_it)

    loss = parallel_model.gam_rho_loss(y_all, parallel_model(x_all))

    make_heatmap(r'likelihood', loss['likelihoods'].detach().cpu().view(len(gamma), len(rho)), "", "", r"$\rho$", r"$\gamma$", args.base_model_path + "plots/likelihood.pdf", save=True, figsize=(8, 6))
    make_heatmap(r'mse', loss['residuals'].pow(2).mean(0).detach().cpu().view(len(gamma), len(rho)), "", "", r"$\rho$", r"$\gamma$", args.base_model_path + "plots/mse.pdf", save=True, figsize=(8, 6))
    make_heatmap(r'wmse', loss['weighted_mse'].detach().cpu().view(len(y_all), len(gamma), len(rho)).mean(0), "", "", r"$\rho$", r"$\gamma$", args.base_model_path + "plots/wmse.pdf", save=True, figsize=(8, 6))
    make_heatmap(r'mean pen', loss['scaled_mean_reg'].detach().cpu().view(len(gamma), len(rho)), "", "", r"$\rho$", r"$\gamma$", args.base_model_path + "plots/scaled_mean_pen.pdf", save=True, figsize=(8, 6))
    make_heatmap(r'prec pen', loss['scaled_prec_reg'].detach().cpu().view(len(gamma), len(rho)), "", "", r"$\rho$", r"$\gamma$", args.base_model_path + "plots/scaled_prec_pen.pdf", save=True, figsize=(8, 6))

    dense_grid = torch.linspace(x_all.min(), x_all.max(), 1000)
    dense_grid = dense_grid.to(device)
    gw = dense_grid[1] - dense_grid[0]

    grad_pens = parallel_model.grad_pen(dense_grid)
    
    make_heatmap(r'mean grad integral', grad_pens['mean_pen'].detach().cpu().view(len(gamma), len(rho)), "", "", r"$\rho$", r"$\gamma$", args.base_model_path + "plots/mean_int.pdf", save=True, figsize=(8, 6))
    make_heatmap(r'prec grad integral', grad_pens['prec_pen'].detach().cpu().view(len(gamma), len(rho)), "", "", r"$\rho$", r"$\gamma$", args.base_model_path + "plots/prec_int.pdf", save=True, figsize=(8, 6))
    
    

if __name__ == '__main__':
    args = get_args()
    
    if args.pre_trained_path is None:
        with open(args.base_model_path + 'commandline_args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        checkpoint = torch.load(args.pre_trained_path)

        with open(args.base_model_path + 'commandline_args_' + str(checkpoint['epoch']) + '.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)


    main(args)