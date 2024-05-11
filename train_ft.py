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
from heteroskedastic_nns.load_args import get_ft_args
from heteroskedastic_nns.parallel_ft import ParallelDFT
from heteroskedastic_nns.datasets.generate_data import prep_synth_data
from heteroskedastic_nns.utils.utils import gam_rho_to_alpha_beta, fit_field_theory, make_heatmap



def main(args):
    print('main')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    het_noise = not args.homoskedastic_noise
    x_all, y_all, tm = prep_synth_data(args.dataset, device, args.data_seed, args.samp_size, het_noise)

    plt.plot(x_all.cpu().detach(), y_all.cpu().detach())
    plt.savefig(args.base_model_path + 'data.png')

    tm = tm.to(device)

    symlog = [1.0, 0.999999, 0.99999, 0.99999, 0.9999, 0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0]
    gamma = symlog[1:-1]
    rho = [i for i in reversed(symlog[1:-1])]

 

    tm = tm[:, None]

    print('ny')
    print(args.noisy_y)
    fitted_ft, loss_stats = fit_field_theory(x=x_all, y=tm, device=device, seed=args.train_seed, gamma=gamma, rho=rho, 
                     max_epochs=args.epochs, lr=args.lr, lr_min=args.lr_min, 
                     lr_max=args.lr_max, cycle_mode=args.cycle_mode, 
                     base_model_path=args.base_model_path, step_size_up=args.step_size_up, opt_scheme=args.opt_scheme, noisy_y=args.noisy_y)
    

    

if __name__ == '__main__':
    args = get_ft_args()

    with open(args.base_model_path + 'commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    main(args)