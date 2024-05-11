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
from heteroskedastic_nns.load_args import get_args
from heteroskedastic_nns.parallel_model import ParallelFF
from heteroskedastic_nns.datasets.generate_data import prep_double_sine, prep_polynomial_data, prep_uci_data
from heteroskedastic_nns.utils.utils import plot_result, plot_sd_res, make_heatmap, plot_parallel_model, gam_rho_to_alpha_beta, run_uci_exp

def main(args):
    print('main')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    x=torch.tensor(np.load('/home/ewongtoi/Documents/heteroskedastic_nns/climate/train_input.npy')).to(device)
    y=torch.tensor(np.load('/home/ewongtoi/Documents/heteroskedastic_nns/climate/train_target.npy')).to(device)

    sub_x = x[::2,:]

    targ_ind = 123
    print('targ ind: ', targ_ind)
    sub_y = y[::2, targ_ind]
    
    print(sub_x.shape)
    print(sub_y.shape)


    if args.diag:
        symlog = [1.0, 0.99999999999, 0.9999999999, 0.999999999, 0.99999999, 0.9999999, 0.999999, 0.99999, 0.9999, 0.999, 0.99] + [sl * .01 + .1 for sl in range(81)] + [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001, 0.00000000001, 0.0]
        gamma = symlog[1:-1]

        rho = [0.5, 0.6]

    if args.beta_nll:
        symlog = [1.0, 1.0, 1.0, 1.0, 1.0]
        gamma = symlog[1:-1]
        rho = symlog[1:-1]

    if args.mle:
        symlog = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        gamma = symlog
        rho = symlog

    if not args.diag and not (args.beta_nll or args.mle):
        symlog = [1.0, 1.0, 0.9999, 0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001, 0.00000000001, 0.0, 0.0]
        #symlog = [1.0, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.0]
        gamma = symlog[1:-1]
        rho = [i for i in reversed(symlog[1:-1])]



    print('ppml')
    print(args.per_param_loss)

    trained_model =  run_uci_exp(sub_x, sub_y, device, args.train_seed, gamma, rho, sub_x.shape[1], 1, 
                                    args.act_func, args.prec_act_func, args.epochs, args.lr_min, args.lr_max, args.cycle_mode, args.base_model_path + 'first/',  
                                    pre_trained_path=args.pre_trained_path, per_param_loss=args.per_param_loss, hidden_layers=args.hidden_layers, hidden_size=args.hidden_size, 
                                    step_size_up=args.step_size_up, clip=args.clip, batch_size=args.batch_size, mean_warmup=args.mean_warmup, diag=args.diag, beta_nll=args.beta_nll, var_param=args.var_param)
    


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