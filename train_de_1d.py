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
from heteroskedastic_nns.load_args import get_fourier_args
from heteroskedastic_nns.datasets.generate_data import prep_double_sine, prep_polynomial_data, prep_uci_data, prep_synth_data
from heteroskedastic_nns.utils.utils import  run_ls_exp

def main(args):
    print('main')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    het_noise = not args.homoskedastic_noise
    x_all, y_all, _ = prep_synth_data(args.dataset, device, args.data_seed, args.samp_size, het_noise)


    torch.manual_seed(args.data_seed)
    inds = torch.randperm(len(x_all))


    plt.scatter(x_all.cpu().detach(), y_all.cpu().detach(), marker='.')

    plt.savefig(args.base_model_path + 'data.png')

    

    pickle.dump(x_all, open(args.base_model_path + 'x_all.p', 'wb'))
    pickle.dump(y_all, open(args.base_model_path + 'y_all.p', 'wb'))



    if args.diag:
        hypervals = [1.0, 0.99999999999, 0.9999999999, 0.999999999, 0.99999999, 0.9999999, 0.999999, 0.99999, 0.9999, 0.999, 0.99] + [sl * .01 + .1 for sl in range(81)] + [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001, 0.00000000001, 0.0]
        gamma = hypervals
        rho = [.5, .6]
    else: 
        #hypervals = [.99999, .9999, .999, .99, .9, .8, .7, .6, .5, .4, .3, .2, .1, .01, .001, .0001, .00001]
        hypervals = [0.9999, 0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001, 0.00000000001]
    
        gamma = hypervals
        rho = [h for h in reversed(hypervals)]


    fail_it, aug_model= run_ls_exp(x_all, y_all, device, args.train_seed, 1, 1,
            args.act_func, args.scale_act_func, args.epochs,
            args.lr_min, args.lr_max, args.cycle_mode, args.base_model_path + 'first/', gammas=gamma, rhos=rho, inv_param=args.inv_param,
            batch_size=args.batch_size, diag=args.diag, B_dim=args.B_dim, B_sigma=args.B_sigma,
            pre_trained_path=args.pre_trained_path, hidden_layers=args.hidden_layers, hidden_size=args.hidden_size,
            step_size_up=args.step_size_up, clip=args.clip, mean_warmup=args.mean_warmup)
    
    '''
    fail_it, aug_model = run_aug_exp(x_all, y_all, device, args.train_seed, gamma, rho, 1, 1,
            args.act_func, args.prec_act_func, args.epochs,
            args.lr_min, args.lr_max, args.cycle_mode, args.base_model_path + 'first/', per_param_loss=args.per_param_loss,
            pre_trained_path=args.pre_trained_path, hidden_layers=args.hidden_layers, hidden_size=args.hidden_size,
            step_size_up=args.step_size_up, clip=args.clip, mean_warmup=args.mean_warmup, aug_x=args.aug_x, aug_y=args.aug_y,
            x_noise_scale=args.x_noise_scale, y_noise_scale=args.y_noise_scale, grad_aware=args.grad_aware)
    '''
    
    print(fail_it)

    dense_x = torch.linspace(x_all.min(), x_all.max(), 300).to(device)
    bvx = x_all.unsqueeze(1).expand(x_all.shape[0], aug_model.num_models, x_all.shape[1])
    #output = aug_model(bvx)

    #means = output['location'][0,:, :].cpu()
    #precs = output['scale'][0,:,:].cpu()

    #ci = precs.pow(-.5).cpu()

    #plt.plot(x_all.cpu(), means.cpu().detach())
    #plt.plot(x_all.cpu(), y_all.cpu())

    #plt.fill_between(dense_x.cpu().flatten(), (means.cpu()-ci.cpu()).flatten(), (means.cpu()+ci.cpu()).flatten(), color='b', alpha=.2)
    #plt.fill_between(dense_x.cpu().flatten(), (means.cpu()-2*ci.cpu()).flatten(), (means.cpu()+2*ci.cpu()).flatten(), color='b', alpha=.1)

    #plt.savefig(args.base_model_path + 'fit.png')

    

    #plot_ls_model(aug_model, bvx, y_all, stats=None, iteration=args.epochs, dense_x=None, path=args.base_model_path, ns_data=None)
    #dense_x = (torch.linspace(x_all.min(), x_all.max())[:, None]).to(device)
    #plot_parallel_model(spff=parallel_model, x_m=x_all, x_p=x_all, y_m=y_all, y_p=y_all, stats=all_loss, iteration=args.epochs, dense_x=dense_x, path=args.base_model_path)

    '''
    for k, curr_dict in loss_dicts.items():

        npts = n_datapoints[k]

        make_heatmap(r'likelihood ' + k, curr_dict['likelihoods'].detach().cpu().view(len(gamma), len(rho)), "", "", r"$\rho$", r"$\gamma$", args.base_model_path + "plots/likelihood_" + k + ".pdf", save=True, figsize=(8, 6))
        

        make_heatmap(r'mse ' + k, curr_dict['residuals'].pow(2).view(npts, len(gamma), len(rho)).mean(0).detach().cpu(), "", "", r"$\rho$", r"$\gamma$", args.base_model_path + "plots/mse_" + k + ".pdf", save=True, figsize=(8, 6))

    make_heatmap(r'mean pen', all_loss['scaled_mean_reg'].detach().cpu().view(len(gamma), len(rho)), "", "", r"$\rho$", r"$\gamma$", args.base_model_path + "plots/scaled_mean_pen" + k + ".pdf", save=True, figsize=(8, 6))
    make_heatmap(r'prec pen', all_loss['scaled_prec_reg'].detach().cpu().view(len(gamma), len(rho)), "", "", r"$\rho$", r"$\gamma$", args.base_model_path + "plots/scaled_prec_pen" + k + ".pdf", save=True, figsize=(8, 6))



    dense_grid = torch.linspace(x_all.min(), x_all.max(), 500)
    dense_grid = dense_grid.to(device)
    gw = dense_grid[1] - dense_grid[0]

    grad_pens = parallel_model.grad_pen(dense_grid)
    
    make_heatmap(r'mean grad integral', grad_pens['mean_pen'].detach().cpu().view(len(gamma), len(rho)), "", "", r"$\rho$", r"$\gamma$", args.base_model_path + "plots/mean_int.pdf", save=True, figsize=(8, 6))
    make_heatmap(r'prec grad integral', grad_pens['prec_pen'].detach().cpu().view(len(gamma), len(rho)), "", "", r"$\rho$", r"$\gamma$", args.base_model_path + "plots/prec_int.pdf", save=True, figsize=(8, 6))
    '''
    

if __name__ == '__main__':
    args = get_fourier_args()
    
    if args.pre_trained_path is None:
        with open(args.base_model_path + 'commandline_args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        checkpoint = torch.load(args.pre_trained_path)

        with open(args.base_model_path + 'commandline_args_' + str(checkpoint['epoch']) + '.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)


    main(args)