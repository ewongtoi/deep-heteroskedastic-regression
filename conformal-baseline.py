import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import numpy as np

from torch.optim.optimizer import Optimizer, required
import dill as pickle

from torch.distributions.normal import Normal

from tqdm import tqdm

import pysgmcmc
from scipy.stats import wasserstein_distance
import sys
import heteroskedastic_nns.datasets.load_uci
#from heteroskedastic_nns.parallel_model import ParallelFF
#from heteroskedastic_nns.model import SingleFF
from heteroskedastic_nns.datasets.generate_data import prep_synth_data, prep_uci_data
from heteroskedastic_nns.utils.utils import plot_result, make_heatmap, plot_parallel_model, plot_split_model, num_grad, vec_num_grad, mult_approx_grad, gam_rho_to_alpha_beta, expected_calibration_error_nfe
from torchquad import MonteCarlo, set_up_backend
import math
from tabulate import tabulate
import copy

from matplotlib.colors import LogNorm, TwoSlopeNorm

# hetero coverage batched

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(224)
dataset="power"
x, y, xt, yt = prep_uci_data(f'{dataset}', device)
sum_sdmse1 = []
sum_mse = []
sum_cov = []

inds = []

big_n = yt.shape[0]
batch_size = 10

with torch.no_grad():
    for i in range(6):
        with open(f'/extra/ucibdl0/ewongtoi/heteroskedastic_nns/conformal-supp/uci/local/{dataset}/run-{i}/first/49999_parallel_model.p', 'rb') as f:
        
            ucimod = (pickle.load(f))

        print(i)
            
        subout = ucimod(x[1::2, :])
        cal_resids1 = (subout['mean'] - y[1::2].unsqueeze(1)).view(y.shape[0] // 2, 101, 2).abs()
        sds1 = subout['precision'].pow(-.5).view(y.shape[0] // 2, 101, 2)
        
        conform_scores = cal_resids1 / sds1
        
        sdmses = (cal_resids1 - sds1).pow(2).mean(0)
        
        quantiles1 = torch.quantile(conform_scores, 0.682, dim=0)


        mn_min_index = torch.argmin(cal_resids1.mean(0), dim=0)
        sd_min_index = torch.argmin(sdmses, dim=0)


        ind0 = (mn_min_index[0] + sd_min_index[0]) // 2 
        ind1 = 0

        
        inds.append((ind0.item(), ind1))
        
        sd_se = 0
        mn_se = 0
        
        batches = big_n // batch_size
        for b in range(batches):
            start = b * batch_size
            end = min((b+1) * batch_size, yt.shape[0])
            
            sub_xt = xt[start:end, :]
            sub_yt = yt[start:end]
            
            
            batch_pred = ucimod(sub_xt)
            
            batch_resids = (batch_pred['mean'] - sub_yt.unsqueeze(1)).view(sub_yt.shape[0], 101, 2).abs()
            
            mn_se += batch_resids[:ind0, ind1].pow(2)
            
            cal_sd = (quantiles1 * batch_pred['precision'].pow(2).view(sub_yt.shape[0], 101, 2))
            sd_se += (batch_resids[:, ind0, ind1] - cal_sd[:, ind0, ind1]).pow(2)
            
            cov_count = (batch_resids < (cal_sd * 1)).float()
        
        
        sum_cov.append(cov_count / big_n)

        sum_sdmse1.append(sd_se / big_n)
        
        sum_mse.append(mn_se / big_n)
        
        del ucimod

    
sdmse_tensor = torch.tensor(sum_sdmse1)
mse_tensor = torch.tensor(sum_mse)
cov_tensor = torch.tensor(sum_cov)

print('sdmse')
print(sdmse_tensor.mean())
print(sdmse_tensor.var().pow(.5))

print('mse')
print(mse_tensor.mean())
print(mse_tensor.var().pow(.5))

print('mean cov')
print(cov_tensor.mean())
print(cov_tensor.var().pow(.5))

print(inds)