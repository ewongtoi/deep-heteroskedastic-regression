#%%
from multiprocessing.context import ForkContext
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt



import numpy as np
import torch, torch.nn as nn

from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import matplotlib.patches as mpatches

from heteroskedastic_nns.datasets.load_uci.uci_loader import UCIDatasets

import sys
import math

from tqdm import tqdm

from torch.utils.data import Dataset, TensorDataset
#### generate data
#%%

class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

def gen_curve_data(n, het=True, seed=4):
    torch.manual_seed(seed) 

    x = (torch.rand((n, 1))-0.5)*3
    
    x = torch.sort(x.squeeze())[0].unsqueeze(-1)
    
    ind = ((torch.arange(0, n)+1)/n)*2+0.1
    
    true_mu = x[:,0] - 2*x[:, 0]**2 + 0.5*x[:,0]**3
    
    if het:
        y = true_mu + torch.randn(n) * ind.abs()
    else: 
        y = true_mu + torch.randn(n)
        
    y = y.unsqueeze(-1)
    
    
    return x, y, true_mu

# generates noisy data over a dense area
def gen_noisy_data(power, n=252, noise=[.1, 1, 3, 10], n_groups=4, seed=1):
    torch.manual_seed(seed)    # reproducible

    noise = [n * (5**(power-1)) for n in noise]
    x = torch.unsqueeze(torch.linspace(-10, 10, n), dim=1) # x data (tensor), shape=(100, 1)


    per_grp = n // n_groups
    noise_pattern = torch.tensor(noise).repeat(per_grp).sort()[0]



    # adds on heteroscedastic noise
    het_noise = torch.normal(mean=torch.zeros(1), std=noise_pattern)
    true_mu = torch.pow(x, power)
    y = true_mu + het_noise[:, None]

    x = x/10

    return(x, y, het_noise, noise_pattern[:, None], true_mu)

def gen_triangle_data(dist, theta=-45):

    # equilateral triangle
    x = torch.tensor([0., dist / 2., dist])
    y = torch.tensor([0., dist * (3**(.5) / 2.), 0.])

    center_x = x.mean()
    center_y = y.mean()

    pts = torch.stack((x-center_x, y-center_y))
    phi = torch.tensor(theta * math.pi / 180)
    
    s = torch.sin(phi)
    c = torch.cos(phi)
    rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

    rot_pts = rot @ pts

    flat_x = rot_pts[0, :].flatten()
    flat_y = rot_pts[0, :].flatten()

    sortinds = flat_x.sort(0)[1]

    flat_x_sort = flat_x[sortinds][:, None]
    flat_y_sort = flat_y[sortinds][:, None]

    

    return(flat_x_sort, flat_y_sort)

def poly_data(model_power, data_power, n=256, noise=[.1, 1, 3, 10], n_groups=4, seed=1, v_noise=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = n

    if v_noise:
        x_all, y_all, het_noise, noise_pattern, true_mu = vary_noise(power=data_power, n=int(n* 2.25))
    else:
        x_all, y_all, het_noise, noise_pattern, true_mu = gen_noisy_data(power=data_power, n=n)
    

    
    return(0)


def vary_noise(power, n=256, noise=[.1, 1, 3, 10], n_groups=4, seed=1):


    x = torch.unsqueeze(torch.linspace(-10, 10, n), dim=1)  # x data (tensor), shape=(100, 1)

    per_grp = n // n_groups
    noise_pattern = torch.tensor(noise).repeat(per_grp).sort()[0]

    # adds on heteroscedastic noise
    het_noise = torch.normal(mean=torch.zeros(1), std=noise_pattern)
    true_mu = torch.pow(x, power)
    y = true_mu + het_noise[:, None]

    cut = n // 4

    first = torch.randperm(cut)[:(cut // 8)]
    second = torch.randperm(cut)[:(cut // 4)] + cut
    third = torch.randperm(cut)[:(cut // 2)] + 2*cut
    fourth = torch.tensor([i for i in range(3*cut, 4*cut)])

    sorted_inds, _ = torch.sort(torch.cat([first, second, third, fourth]))
    
    x = x/10

    return(x[sorted_inds], y[sorted_inds], het_noise[sorted_inds], (noise_pattern[sorted_inds])[:, None], true_mu)

#%%

def prep_uci_data(uci_dataset, device, standardize=True):
    np.random.seed(224)
    dataset = UCIDatasets(uci_dataset, data_path="/home/ewongtoi/Documents/heteroskedastic_nns/heteroskedastic_nns/")

    N = dataset.data.shape[0]
    torch.manual_seed(4)
    rand_inds = torch.randperm(N)

    cut = dataset.data.shape[0] // 3

    # holds 1/3 thru 3/3 of the data
    x = torch.tensor(dataset.data[rand_inds[cut:N],:dataset.in_dim]).float().to(device)
    y = torch.tensor(dataset.data[rand_inds[cut:N],dataset.in_dim:]).float().to(device)

    if standardize:
        x_means = x.mean(dim=0, keepdim=True)
        x_stds = x.std(dim=0, keepdim=True)

        y_means = y.mean()
        y_stds = y.std()

        x = (x - x_means) / x_stds
        y = (y - y_means) / y_stds

    # holds 0/3 thru 1/3 of the data
    x_test = torch.tensor(dataset.data[rand_inds[0:cut],:dataset.in_dim]).float().to(device)
    y_test = torch.tensor(dataset.data[rand_inds[0:cut],dataset.in_dim:]).float().to(device)

    if standardize:
        x_test = (x_test - x_means) / x_stds
        y_test = (y_test - y_means) / y_stds


    x = x.to(device)
    y = y.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    return(x, y, x_test, y_test)

def prep_polynomial_data(samp_size, power, device, v_noise=False, basis_dim=None, seed=1):

    N = samp_size

    if v_noise:
        x_all, y_all, het_noise, noise_pattern, true_mu = vary_noise(power=power, n=int(samp_size * 2.25))
    else:
        x_all, y_all, het_noise, noise_pattern, true_mu = gen_noisy_data(power=power, n=samp_size, seed=seed)

    x_all = x_all.flatten()
    y_all = y_all.flatten() / torch.tensor([10]).pow(power)

    rand_inds = torch.randperm(samp_size)
    
    cut = N // 3

    x = torch.tensor(x_all[rand_inds[cut:N]]).float().to(device)
    y = torch.tensor(y_all[rand_inds[cut:N]]).float().to(device)


    
    aaa, reord = torch.sort(x)

    x = x[reord].unsqueeze(1)
    y = y[reord].unsqueeze(1)


    n = len(y)
    train_data = []

    for i in range(n):
        train_data.append([x[i], y[i]])

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=n)

    

    x_test = torch.tensor(x_all[rand_inds[0:cut]]).float().to(device)
    y_test = torch.tensor(y_all[rand_inds[0:cut]]).float().to(device)

    aaa, reord_test = torch.sort(x_test)

    x_test = x_test[reord_test].unsqueeze(1)
    y_test = y_test[reord_test].unsqueeze(1)

    x = x.to(device)
    y = y.to(device)

    x_test.to(device)
    y_test.to(device)

    # expand basis of the data; don't include intercept (that's in the model already)
    if basis_dim is not None:
        x = torch.squeeze(torch.stack([x.pow(i) for i in range(1, (basis_dim+1))]).T)

        x_test = torch.squeeze(torch.stack([x_test.pow(i) for i in range(1, (basis_dim+1))]).T)

        if basis_dim == 1:
            x = x[:, None]
            x_test = x[:, None]
        

    return(x, y, x_test, y_test, noise_pattern[reord], noise_pattern[reord_test], true_mu)


def prep_double_sine(nn_target, n_mean_cycles, n_noise_cycles, extra_ends, seed, het=True):
    torch.manual_seed(seed) 
    lb=0.
    ub=1.

    xorg_init = torch.unsqueeze(torch.linspace(lb, ub, nn_target), dim=1)
    gw_init = xorg_init[1]-xorg_init[0]


    nn_aug = nn_target + extra_ends * 2
    xorg_aug = torch.unsqueeze(torch.linspace((lb - gw_init * extra_ends).item(), (ub + gw_init* extra_ends).item(), nn_aug), dim=1)
    xorg_init == xorg_aug[extra_ends:-extra_ends]


    ind = (((torch.arange(0, nn_target)+1)/nn_target))

    true_mu = 2 * torch.sin(n_mean_cycles * 2* torch.pi * xorg_aug[:,0])
    if het:
        mal = torch.sin(n_noise_cycles * 2 *torch.pi*xorg_aug[:,0]) + 1.25
    else:
        mal = torch.ones_like(xorg_aug[:, 0]) * 1

    sy = (true_mu + (mal*torch.randn(len(mal)))).unsqueeze(-1)



    return(xorg_init, xorg_aug, ind, true_mu, mal, sy)

def prep_synth_data(dataset, device, seed=1, sample_size=64, het=True):
    # sine data/noise
    tm = None
    if dataset == "sine" and het:
        _, xorg_aug, _, _, _, sy= prep_double_sine(sample_size, 2, 3, 1, seed)
        x = xorg_aug.to(device)
        y_raw = sy.to(device)
        
        y_mean = y_raw.mean()
        y_std = y_raw.var().pow(.5)

        y = (y_raw - y_mean) / y_std

    elif dataset == "sine" and not het:
        _, xorg_aug, _, _, _, sy= prep_double_sine(sample_size, 2, 0, 1, seed, False)
        x = xorg_aug.to(device)
        y_raw = sy.to(device)
        
        y_mean = y_raw.mean()
        y_std = y_raw.var().pow(.5)

        y = (y_raw - y_mean) / y_std

    elif dataset == "curve" and het:
        x, y_raw, tm = gen_curve_data(sample_size, True, seed)

        y_mean = y_raw.mean()
        y_std = y_raw.var().sqrt()

        y = (y_raw - y_mean) / y_std

        x=x.to(device)
        y=y.to(device)

    elif dataset == "curve" and not het:
        x, y_raw, tm = gen_curve_data(sample_size, False, seed)

        y_mean = y_raw.mean()
        y_std = y_raw.var().sqrt()

        y = (y_raw - y_mean) / y_std
        
        x=x.to(device)
        y=y.to(device)

    # polynomial data
    elif dataset.isdigit() and het:
        power = int(dataset)
        x, y, x_test, y_test, noise_pattern_train, noise_pattern_test, tm = prep_polynomial_data(sample_size, power, device, seed=seed)


        x_all, ord_inds = torch.concat([x, x_test]).sort(0)
        y_all = torch.concat([y, y_test])[ord_inds].flatten()[:, None]

        y_mean = y_all.mean()
        y_std = y_all.var().pow(.5)

        y_all = (y_all - y_mean) / y_std

        x = x_all
        y = y_all
    
    elif dataset.isdigit() and not het:
        torch.manual_seed(seed)
        power = int(dataset)
        x = torch.unsqueeze(torch.linspace(-1, 1, sample_size), dim=1)
        y_raw = x.pow(power) + torch.normal(mean=torch.zeros_like(x), std=.1)
        y_mean = torch.mean(y_raw)
        y_std = y_raw.var().pow(.5)
        y = (y_raw - y_mean) / y_std

        x = x.to(device)
        y = y.to(device)
    
    
    else:
        return 0
    
    return x, y, tm

# %%

