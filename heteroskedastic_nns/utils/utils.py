import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import matplotlib.pyplot as plt


from tqdm import tqdm
import dill as pickle
import warnings
from heteroskedastic_nns.parallel_model import ParallelFF
from heteroskedastic_nns.parallel_ft import ParallelDFT

from matplotlib.ticker import MaxNLocator
import seaborn as sns
import math
from matplotlib.colors import SymLogNorm


mse = torch.nn.MSELoss()



def mixture_chunk_prediction(preds, n_chunks):

        raw_means = torch.chunk(preds["mean"], chunks=n_chunks, dim=1)

        raw_vars = torch.chunk(preds["precision"].pow(-1), chunks=n_chunks, dim=1)

        all_means = torch.zeros_like(raw_means[0])
        all_precs = torch.zeros_like(raw_vars[0])

        # output shape is n datapoints, n models, n dim
        for ch in range(n_chunks):
            mu_bar = raw_means[ch].mean(dim=1)
            
            mean_sq_bar = raw_means[ch].pow(2).mean(dim=1)

            var_bar = raw_vars[ch].mean(dim=1)

            prec = (var_bar + mean_sq_bar - mu_bar.pow(2)).pow(-1)

            all_means[:, ch, :] = mu_bar
            all_precs[:, ch, :] = prec

        return {
            "mean": all_means,
            "precision": all_precs
        }

def ci_obs_avg(x, y, model, tau, device, prec_param=True, mixture=1):

    # Create a standard normal distribution tensor
    normal_distribution = torch.distributions.Normal(0, 1)

    # Get the quantile
    
    alpha = 1 - ((1-tau) / 2)
    quantile = normal_distribution.icdf(torch.tensor(alpha)).to(device)
    
    model = model.to(device)

    if mixture > 1:
        preds = mixture_chunk_prediction(model(x), n_chunks=mixture)
    else:
        preds = model(x)

    
    # len(y) x num_models x 1
    uq_taus = preds['mean'] + quantile * preds['precision'].pow(-.5)
    lq_taus = preds['mean'] - quantile * preds['precision'].pow(-.5)
 
    
    y = y.view(len(y), 1, 1)
    y_expanded = y.repeat(1, uq_taus.shape[1], 1)
   
    # returns num_models x 1 tensor where each is the proportion of data pts 
    # the model covered
    return ((uq_taus > y_expanded) * (lq_taus < y_expanded)).float().mean(dim=0)#, uq_taus, lq_taus, preds['mean']




def p_obs_avg(x, y, model, tau, device, prec_param=True, mixture=1):

    # Create a standard normal distribution tensor
    normal_distribution = torch.distributions.Normal(0, 1)

    # Get the quantile
    quantile = normal_distribution.icdf(torch.tensor(tau)).to(device)

    if mixture > 1:
        preds = mixture_chunk_prediction(model(x), n_chunks=mixture)
    else:
        preds = model(x)

    
    # len(y) x num_models x 1
    q_taus = preds['mean'] + quantile * preds['precision'].pow(-.5)
 
    
    y = y.view(len(y), 1, 1)
    y_expanded = y.repeat(1, q_taus.shape[1], 1)
    
    # returns num_models x 1 tensor where each is the proportion of data pts 
    # the model covered
    return (q_taus > y_expanded).float().mean(dim=0)

def expected_calibration_error(x, y, model, device, samples=1000, mixture=1):
    ece = 0
    
    # samples is the number of taus to draw
    for s in range(samples):
        tau = torch.rand(1).to(device)
        ece += (p_obs_avg(x, y, model, tau, device, mixture=mixture) - tau).abs()
    
    return ece / samples

def expected_ci_coverage(x, y, model, device, samples=1000, mixture=1):
    eci = 0
    
    
    # samples is the number of taus to draw
    for s in range(samples):
        tau = torch.rand(1).to(device)
        eci += (ci_obs_avg(x.to(device), y.to(device), model, tau, device, mixture=mixture) - tau).abs()
        
        #if s % (samples// 10) == 0:
        #    print(s)
    
    return eci / samples


def p_obs_avg_nfe(y, model, tau, device, prec_param=True, mixture=False):

    # Create a standard normal distribution tensor
    normal_distribution = torch.distributions.Normal(0, 1)

    # Get the quantile
    quantile = normal_distribution.icdf(torch.tensor(tau)).to(device)

    model.log_lam_stack.pow(-.5)
    q_taus = model.mu_stack + quantile * model.log_lam_stack.exp().pow(-.5)

    

    return (q_taus > y.expand(q_taus.shape[0], q_taus.shape[1]).unsqueeze(2)).float().mean(dim=0)

def expected_calibration_error_nfe(y, model, device, samples=1000):
    ece = 0
    
    for s in range(samples):
        tau = torch.rand(1).to(device)
        ece += (p_obs_avg_nfe(y, model, tau, device) - tau).abs()
    
    return ece / samples


# performs linear interpolation (assuming even spacing)
def average_neighbors(tensor):
    left = tensor[:-1]  # Elements from the beginning to the second-to-last
    right = tensor[1:]  # Elements from the second to the last to the end
    average = (left + right) / 2
    return average


def gam_rho_to_alpha_beta(gamma, rho):
    gamma = torch.tensor(gamma)
    rho = torch.tensor(rho)
    alpha = ((1-rho)*gamma) / rho
    beta = ((1-rho) * (1-gamma)) / rho
    
    return alpha, beta


def mult_approx_grad(point, model, device, eps=0.0001):
    repeated = point.repeat(point.shape[1], 1).to(device)
    
    eps_mat = (torch.eye(point.shape[1]) * eps).to(device)
    
    front_shift = repeated + eps_mat
    back_shift = repeated - eps_mat
    with torch.no_grad():
      mplus = model(front_shift)
      mminus = model(back_shift)
    
    mean_grad = (mplus['mean'] - mminus['mean']) / (2 * eps)
    prec_grad = (mplus['precision'] - mminus['precision']) / (2 * eps)
    
    return(mean_grad, prec_grad)


def stack_closed_form_mu(lam_stack, y, alpha, device):
    # eltwise mult (think reweight)
    n = lam_stack.size()[0]

    wt_y = ((lam_stack * y[:, None, :]).squeeze().T).to(device)
  
    lapmat = torch.diag_embed(torch.ones(n), -1)[0:n, 0:n] - 2 * torch.diag(torch.ones(n)) + torch.diag_embed(torch.ones(n), 1)[0:n, 0:n]

    # wrap the boundaries
    #lapmat[0, -1] = 1
    #lapmat[-1, 0] = 1  
    
    lapmat_stack = lapmat.repeat(lam_stack.shape[1], 1, 1).to(device)
    diag_lam = torch.diag_embed(lam_stack.squeeze(-1).T).to(device)
    

    
    # ret inv( diag(lam) + alpha/2 lapmat ) * lam * y // sign flip before the return because of the laplace matrix
    return (torch.linalg.solve(-(2*alpha)[:, None, None] * lapmat_stack + diag_lam,  wt_y)).transpose_(0, 1)[:, :, None]



def fit_field_theory(x, y, device, seed, gamma, rho, 
            max_epochs, lr, lr_min, lr_max, cycle_mode, base_model_path, step_size_up=1000, opt_scheme=None, noisy_y=False):
    print('re')


    epochs = max_epochs
    
    torch.manual_seed(seed)

    start_offset = 0

    dft = ParallelDFT(grid_discretization=x, gammas=gamma, rhos=rho, btw_pts=None, init_loc=0.).to(device)
    
    alphas, _ = gam_rho_to_alpha_beta(dft.gammas, dft.rhos)

    epochs = max_epochs

    if opt_scheme is None:
      optimizer = torch.optim.Adam([dft.mu_stack, dft.log_lam_stack], lr=lr)
      scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr_min, max_lr=lr_max, mode=cycle_mode, cycle_momentum=False, step_size_up=step_size_up)

      losses = []

      for index in tqdm(range(epochs)):
          optimizer.zero_grad()
          if noisy_y:
             noise = torch.randn_like(y) * x.pow(2)
             y_noisy = y + noise
          else:
             y_noisy = y
          loss = dft.gamma_rho_const_noise_integral_loss(y_noisy)
          loss['loss'].sum().backward()
          optimizer.step()
          scheduler.step()

          if index % (epochs // 10) == 0:
              losses.append(loss)
              print(index / epochs)
    
    elif opt_scheme == "split":
      dft = ParallelDFT(grid_discretization=x, gammas=gamma, rhos=rho, btw_pts=None, init_loc=0., split_train=True, split_ratio=0.5).to(device)
      optimizer = torch.optim.Adam([dft.mu_stack, dft.log_lam_stack], lr=lr)
      scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr_min, max_lr=lr_max, mode=cycle_mode, cycle_momentum=False, step_size_up=step_size_up)

      losses = []

      for index in tqdm(range(epochs)):
          if noisy_y:
             noise = torch.randn_like(y) * x.pow(2)
             y_noisy = y + noise
          else:
             y_noisy = y

          optimizer.zero_grad()
          loss = dft.gamma_rho_split_loss(y_noisy)
          loss['loss'].sum().backward(retain_graph=True)
          optimizer.step()
          scheduler.step()

          if index % (epochs // 10) == 0:
              losses.append(loss)
              print(index / epochs)
       

    elif opt_scheme == "closedmu":
      cycles = epochs // (step_size_up * 2)
      optimizer = torch.optim.Adam([dft.log_lam_stack], lr=lr)
      scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr_min, max_lr=lr_max, mode=cycle_mode, cycle_momentum=False, step_size_up=step_size_up)

      losses = []

      index = 0 
      for c in tqdm(range(cycles)):
          if noisy_y:
             noise = torch.randn_like(y) * x.pow(2)
             y_noisy = y + noise
          else:
             y_noisy = y
          # update mu
          print(dft.log_lam_stack.exp().shape)
          print(y.shape)
          print(alphas.shape)
          dft.mu_stack.data = stack_closed_form_mu(dft.log_lam_stack.exp(), y_noisy, alphas, device)


          for j in range(2 * step_size_up):
          # update log lambda
            optimizer.zero_grad()
            loss = dft.gamma_rho_integral_loss(y_noisy)
            loss['loss'].sum().backward()
            optimizer.step()
            scheduler.step()

            if index % (epochs // 10) == 0:
                losses.append(loss)
                print(index / epochs)
            
            index += 1
        


    pickle.dump(dft, open(base_model_path + str(index + start_offset) + '_parallel_dft.p', 'wb'))
    pickle.dump(losses, open(base_model_path + str(index + start_offset) + '_parallel_loss_stats.p', 'wb'))

    return dft, losses
    
    
    

def run_exp(x, y, device, seed, gammas, rhos, 
            n_feature, n_output, act_func, prec_act_func, max_epochs,
            lr, lr_min, lr_max, cycle_mode, base_model_path, per_param_loss=True,
            pre_trained_path=None,
            hidden_size=128, hidden_layers=2, step_size_up=1000, clip=1000,
            mean_warmup=20000, mean_log=True, plots=True, 
            beta_nll=False, diag=False, var_param=False):

    fail_it = -1

    torch.manual_seed(seed)

    start_offset = 0
    keep_keys = ['loss', 'losses', 'mse', 'log_precision', 'raw_mean_reg', 'raw_prec_reg']

    hidden_sizes = [hidden_size for _ in range(hidden_layers)]

    if pre_trained_path is None:
        ppm = ParallelFF(n_feature, n_output, hidden_sizes=hidden_sizes, gammas=gammas, rhos=rhos, activation_func=act_func, precision_activation_func=prec_act_func, per_param_loss=per_param_loss, var_param=var_param, diag=diag)
    else:
        ppm = ParallelFF(n_feature, n_output, hidden_sizes=hidden_sizes, gammas=gammas, rhos=rhos, activation_func=act_func, precision_activation_func=prec_act_func, per_param_loss=per_param_loss, var_param=var_param, diag=diag)
        checkpoint = torch.load(pre_trained_path)
        ppm.load_state_dict(checkpoint['model_state_dict'])
        ppm.train()

        start_offset = checkpoint['epoch'] + 1 # correct for off by one

    ppm = ppm.to(device)
    
    failed_models = [[] for _ in range(ppm.num_models)]

    epochs = max_epochs

    train_stats = []
    grad_ints = []

    
    opt = torch.optim.Adam(ppm.parameters(), lr=lr_max)

    scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=lr_min, max_lr=lr_max, mode=cycle_mode, cycle_momentum=False, step_size_up=step_size_up)


    if pre_trained_path is not None:
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
                
    dense_x = torch.linspace(x.min(), x.max(), 1000)

    dense_x = dense_x.to(device)

    gw = dense_x[1]-dense_x[0]

    for i in tqdm(range(epochs)):
        opt.zero_grad()
        
        if i < (mean_warmup):
          if beta_nll:
            stats = ppm.beta_nll_loss(y, ppm(x))
          else:
            stats = ppm.mean_gam_rho_loss(y, ppm(x))#
        else:
          if beta_nll:
            stats = ppm.beta_nll_loss(y, ppm(x))
          else:
            stats = ppm.gam_rho_loss(y, ppm(x)) 




        loss =  stats['loss']

          # log stats every 2%
        if i % (epochs // 50) == 0:
          sub_stats = {key: stats[key] for key in keep_keys}

          train_stats.append(sub_stats)


        if i == (mean_warmup-1) and mean_log:
            plot_dense_x = torch.linspace(x.min().item(), x.max().item(), 300)[:, None]
            plot_dense_x = plot_dense_x.to(device)
            if plots:
              plot_parallel_model(ppm=ppm, x=x, y=y, stats=train_stats, iteration=i + start_offset, dense_x=plot_dense_x, path=base_model_path)
            
            pickle.dump(grad_ints, open(base_model_path + str(i + start_offset) + '_grad_ints.p', 'wb'))
            pickle.dump(train_stats, open(base_model_path + str(i + start_offset) + '_train_stats.p', 'wb'))
            pickle.dump(ppm, open(base_model_path + str(i + start_offset) + '_parallel_model.p', 'wb'))
            pickle.dump(failed_models, open(base_model_path + str(i + start_offset) + '_failed_models.p', 'wb'))

            PATH = base_model_path + 'full_checkpoint_epochs_' + str(i + start_offset) + '.pt'

            torch.save({
                        'epoch': i + start_offset,
                        'model_state_dict': ppm.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        }, PATH)


        # early termination if breaks
        if loss.isnan() or loss.isinf():
            fail_it = i + start_offset
            
            for j, l in enumerate(stats['losses']):
                if l.isnan() or l.isinf():
                    # record which model and when 
                    failed_models[j].append(i)

            PATH = base_model_path + 'checkpoints_broken/checkpoint_' + str(i + start_offset) + '.pt'

            torch.save({
                        'epoch': i + start_offset,
                        'model_state_dict': ppm.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        }, PATH)
            

            break
                    

        loss.backward()

        torch.nn.utils.clip_grad_norm_(ppm.parameters(), clip)
        opt.step()
        scheduler.step()



    pickle.dump(grad_ints, open(base_model_path + str(i + start_offset) + '_grad_ints.p', 'wb'))
    pickle.dump(train_stats, open(base_model_path + str(i + start_offset) + '_train_stats.p', 'wb'))
    pickle.dump(ppm, open(base_model_path + str(i + start_offset) + '_parallel_model.p', 'wb'))
    pickle.dump(failed_models, open(base_model_path + str(i + start_offset) + '_failed_models.p', 'wb'))

    PATH = base_model_path + 'full_checkpoint_epochs_' + str(i + start_offset) + '.pt'

    plot_dense_x = torch.linspace(x.min().item(), x.max().item(), 300)[:, None]
    plot_dense_x = plot_dense_x.to(device)
    if plots:
      plot_parallel_model(ppm=ppm, x=x, y=y, stats=train_stats, iteration=i + start_offset, dense_x=plot_dense_x, path=base_model_path)   


    print(PATH)
    torch.save({
                'epoch': i + start_offset,
                'model_state_dict': ppm.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, PATH)
    
    return fail_it, ppm


def run_uci_exp(x, y, device, seed, gammas, rhos, 
            n_feature, n_output, act_func, prec_act_func, max_epochs, lr_min, lr_max, 
            cycle_mode, base_model_path, per_param_loss=True, pre_trained_path=None,
            hidden_size=128, hidden_layers=2, step_size_up=1000, clip=1000, batch_size=None, mean_warmup=1000, 
            beta_nll=False, var_param=False, diag=False):
    fail_it = -1

    torch.manual_seed(seed)

    start_offset = 0

    hidden_sizes = [hidden_size for _ in range(hidden_layers)]

    if pre_trained_path is None:
        ppm = ParallelFF(n_feature, n_output, hidden_sizes=hidden_sizes, gammas=gammas, rhos=rhos, activation_func=act_func, precision_activation_func=prec_act_func, per_param_loss=per_param_loss, var_param=var_param, diag=diag)
    else:
        ppm = ParallelFF(n_feature, n_output, hidden_sizes=hidden_sizes, gammas=gammas, rhos=rhos, activation_func=act_func, precision_activation_func=prec_act_func, per_param_loss=per_param_loss, var_param=var_param, diag=diag)
        checkpoint = torch.load(pre_trained_path)
        ppm.load_state_dict(checkpoint['model_state_dict'])
        ppm.train()

        start_offset = checkpoint['epoch'] + 1 # correct for off by one

    ppm = ppm.to(device)
    
    failed_models = [[] for _ in range(ppm.num_models)]

    epochs = max_epochs

    train_stats = []
    grad_ints = []

    keep_keys = ['loss', 'losses', 'mse', 'log_precision', 'raw_mean_reg', 'raw_prec_reg']

    opt = torch.optim.Adam(ppm.parameters(), lr=lr_max)

    scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=lr_min, max_lr=lr_max, mode=cycle_mode, cycle_momentum=False, step_size_up=step_size_up)


    if pre_trained_path is not None:
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
                
    dense_x = torch.linspace(x.min(), x.max(), 1000)

    dense_x = dense_x.to(device)

    gw = dense_x[1]-dense_x[0]

    if batch_size is None:
      for i in tqdm(range(epochs)):
          opt.zero_grad()
          
          if i < mean_warmup:
            if beta_nll:
              stats = ppm.beta_nll_loss(y, ppm(x))
            else:
              stats = ppm.mean_gam_rho_loss(y, ppm(x))
          else:
            if beta_nll:
              stats = ppm.beta_nll_loss(y, ppm(x))
            else:
              stats = ppm.gam_rho_loss(y, ppm(x))



          loss =  stats['loss'] 
            # log stats every 2%
          if i % (epochs // 50) == 0:
              sub_stats = {key: stats[key] for key in keep_keys}

              train_stats.append(sub_stats)
              #grad_ints.append(grad_int)

          if i == mean_warmup-1:
            pickle.dump(grad_ints, open(base_model_path + str(i + start_offset) + '_grad_ints.p', 'wb'))
            pickle.dump(train_stats, open(base_model_path + str(i + start_offset) + '_train_stats.p', 'wb'))
            pickle.dump(ppm, open(base_model_path + str(i + start_offset) + '_parallel_model.p', 'wb'))
            pickle.dump(failed_models, open(base_model_path + str(i + start_offset) + '_failed_models.p', 'wb'))

            PATH = base_model_path + 'half_checkpoint_epochs_' + str(i + start_offset) + '.pt'

            torch.save({
                        'epoch': i + start_offset,
                        'model_state_dict': ppm.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        }, PATH)


          # early termination if breaks
          if loss.isnan() or loss.isinf():
              fail_it = i + start_offset
              
              for j, l in enumerate(stats['losses']):
                  if l.isnan() or l.isinf():
                      # record which model and when 
                      failed_models[j].append(i)

              PATH = base_model_path + 'checkpoints_broken/checkpoint_' + str(i + start_offset) + '.pt'

              torch.save({
                          'epoch': i + start_offset,
                          'model_state_dict': ppm.state_dict(),
                          'optimizer_state_dict': opt.state_dict(),
                          'scheduler_state_dict': scheduler.state_dict(),
                          }, PATH)
              

              break
                      

          loss.backward()

          torch.nn.utils.clip_grad_norm_(ppm.parameters(), clip)
          opt.step()
          scheduler.step()
    else:
      num_batches = x.shape[0] // batch_size
      running_losses = []
      for i in tqdm(range(epochs)):
          running_loss = 0
          for b in range(num_batches):
            start_ind = b * batch_size
            end_ind = min((b + 1) * batch_size, x.shape[0])

            batch_x = x[start_ind:end_ind, :]
            batch_y = y[start_ind:end_ind]

            opt.zero_grad()
            
            if i < (epochs / 2):
              if beta_nll:
                stats = ppm.beta_nll_loss(batch_y, ppm(batch_x))
              else:
                stats = ppm.mean_gam_rho_loss(batch_y, ppm(batch_x))
            else:
              if beta_nll:
                stats = ppm.beta_nll_loss(batch_y, ppm(batch_x))
              else:
                stats = ppm.gam_rho_loss(batch_y, ppm(batch_x))

            loss =  stats['loss'] 

            # early termination if breaks
            if loss.isnan() or loss.isinf():
                fail_it = i + start_offset
                
                for j, l in enumerate(stats['losses']):
                    if l.isnan() or l.isinf():
                        # record which model and when 
                        failed_models[j].append(i)

                PATH = base_model_path + 'checkpoints_broken/checkpoint_' + str(i + start_offset) + '.pt'

                torch.save({
                            'epoch': i + start_offset,
                            'model_state_dict': ppm.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            }, PATH)
                

                break
                        
            loss.backward()

            torch.nn.utils.clip_grad_norm_(ppm.parameters(), clip)
            opt.step()
            scheduler.step()

            running_loss += loss.item()

          # log stats every 2%
          log_freq = (epochs // 50) if (epochs > 50)  else 1

          if i % log_freq == 0:
              sub_stats = {key: stats[key] for key in keep_keys}

              train_stats.append(sub_stats)
              running_losses.append(running_loss)

          if i == (epochs // 2)-1:
            pickle.dump(grad_ints, open(base_model_path + str(i + start_offset) + '_grad_ints.p', 'wb'))
            pickle.dump(train_stats, open(base_model_path + str(i + start_offset) + '_train_stats.p', 'wb'))
            pickle.dump(ppm, open(base_model_path + str(i + start_offset) + '_parallel_model.p', 'wb'))
            pickle.dump(failed_models, open(base_model_path + str(i + start_offset) + '_failed_models.p', 'wb'))

            PATH = base_model_path + 'half_checkpoint_epochs_' + str(i + start_offset) + '.pt'

            torch.save({
                        'epoch': i + start_offset,
                        'model_state_dict': ppm.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        }, PATH)

      pickle.dump(running_losses, open(base_model_path + str(i + start_offset) + '_running_losses.p', 'wb'))
        

    pickle.dump(grad_ints, open(base_model_path + str(i + start_offset) + '_grad_ints.p', 'wb'))
    pickle.dump(train_stats, open(base_model_path + str(i + start_offset) + '_train_stats.p', 'wb'))
    pickle.dump(ppm, open(base_model_path + str(i + start_offset) + '_parallel_model.p', 'wb'))
    pickle.dump(failed_models, open(base_model_path + str(i + start_offset) + '_failed_models.p', 'wb'))

    PATH = base_model_path + 'full_checkpoint_epochs_' + str(i + start_offset) + '.pt'



    print(PATH)
    torch.save({
                'epoch': i + start_offset,
                'model_state_dict': ppm.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, PATH)
    
    return fail_it, ppm





def grad_est(model, pt, eps, device):
    x_pre = pt - eps
    x_post = pt + eps
    
    pred_pre = model(x_pre)
    pred_post = model(x_post)
    
    return {"mgrad": (pred_post['mean'] - pred_pre['mean']) / (2 * eps), "pgrad": (pred_post['precision'] - pred_pre['precision']) / (2 * eps)}


def plot_result(ax, x, y, true_mu, x_plot, mup, sdp):
  ax.scatter(x.squeeze().cpu().detach(), y.squeeze().cpu(), color="tab:blue", marker='.')
  ax.plot(x_plot.squeeze().cpu().detach(), true_mu.squeeze().cpu(), '--', color="black")

  #axa = ax.twinx()
  #if m is not None:
  #  ax.title('mn: ' + str(m.mean_prior_std) + '; std: ' + str(m.var_prior_std))  
  #ax.plot(x_plot.squeeze().detach(), tup[0].squeeze().detach(), color="tab:blue")
  #ax.plot(x_plot.squeeze().detach(), tup[1].squeeze().detach(), color="tab:orange")
  for mu_, std_ in zip(mup, sdp):
    ax.plot(x_plot.squeeze().cpu().detach(), mu_.squeeze().cpu().detach(), color="tab:orange")
    #axa.plot(x_plot.squeeze().detach(), std_.squeeze().detach(), color="tab:orange")


def plot_sd_res(ax, x, res, x_plot, sd_plot):
  ax.scatter(x.squeeze().cpu().detach(), res.squeeze().cpu().detach(), color="tab:blue", marker='.')
  
  for sd in sd_plot:
    ax.plot(x_plot.squeeze().cpu().detach(), sd.squeeze().cpu().detach(), color="tab:orange")



def plot_parallel_model(ppm, x, y, stats, iteration, dense_x=None, path=None, show_plots=False, bound_mn_y=None, bound_sd_y=None, laplace=False):
  gammas = ppm.unique_gammas
  rhos = ppm.unique_rhos

  pts_x = x.sort()[0]
  cts_x = pts_x
  if dense_x is not None:
     cts_x = dense_x.sort()[0]
    
  plot_loss = stats is not None

  if plot_loss:
    loss_grid = torch.stack([l['losses'] for l in stats]).view(len(stats), len(gammas), len(rhos)).cpu().detach() 

  a_labs = ppm.gammas.view(len(gammas), len(rhos)).cpu().detach() 
  b_labs = ppm.rhos.view(len(gammas), len(rhos)).cpu().detach() 


  size=4

  fig_res_sds, axs_res_sds = plt.subplots(len(gammas), len(rhos), figsize=(len(gammas)*size, len(rhos)*size), sharex=True, sharey=False)
  fig_res_sds.tight_layout(pad=2.75)

  fig_mn, axs_mn = plt.subplots(len(gammas), len(rhos), figsize=(len(gammas)*size, len(rhos)*size), sharex=True, sharey=False)
  fig_mn.tight_layout(pad=2.75)

  if plot_loss:
    fig_loss, axs_loss = plt.subplots(len(gammas), len(rhos), figsize=(len(gammas)*size, len(rhos)*size), sharex=True, sharey=False)
    fig_loss.tight_layout(pad=2.75)
    axs_dict = {"res_sds":axs_res_sds, 
              "mns": axs_mn,
              "loss": axs_loss}
  else:
    axs_dict = {"res_sds":axs_res_sds, 
              "mns": axs_mn}


  # complete pass, strip out means, precisions (transformed)
  plot_vals = ppm(pts_x)
  pm_mns = plot_vals['mean']
  pm_mns = pm_mns.view((len(pts_x), len(gammas), len(rhos))).cpu().detach() 
  pm_sds = plot_vals['precision']
  pm_sds = pm_sds.view((len(pts_x), len(gammas), len(rhos))).pow(-.5).cpu().detach() 
  
  cts_plot_vals = ppm(cts_x)
  cts_pm_mns = cts_plot_vals['mean']
  cts_pm_mns = cts_pm_mns.view((len(cts_x), len(gammas), len(rhos))).cpu().detach() 
  cts_pm_sds = cts_plot_vals['precision']
  cts_pm_sds = cts_pm_sds.view((len(cts_x), len(gammas), len(rhos))).pow(-.5).cpu().detach() 


  x_plot = x.cpu().detach().flatten()
  cts_x_plot = cts_x.cpu().detach().flatten()
  y_plot = y.cpu().detach().flatten()

  # each value of reg for the mean network
  for i in range(len(gammas)):

      # each value of reg for the prec/sd network
      for j in range(len(rhos)):
          

          if j == 0:
              for _, axs in axs_dict.items():
                  axs[i][j].set_ylabel(r"alpha: {:.2E}".format(a_labs[i][0]))

          if i == len(gammas)-1:
              for _, axs in axs_dict.items():
                  axs[i][j].set_xlabel(r"beta: {:.2E}".format(b_labs[0][j]))

          
          mns = pm_mns[:, i, j]
          sds = pm_sds[:, i, j]

          cts_mns = cts_pm_mns[:, i, j]
          cts_sds = cts_pm_sds[:, i, j]

          resids = (mns - y_plot).abs()

          axs_dict['mns'][i][j].fill_between(cts_x_plot.cpu().flatten(), (cts_mns-cts_sds).flatten(), (cts_mns+cts_sds).flatten(), color='b', alpha=.2)
          axs_dict['mns'][i][j].fill_between(cts_x_plot.cpu().flatten(), (cts_mns-2*cts_sds).flatten(), (cts_mns+2*cts_sds).flatten(), color='b', alpha=.1)

          axs_dict['mns'][i][j].scatter(x_plot, y_plot)
          axs_dict['mns'][i][j].plot(cts_x_plot, cts_mns, c='tab:orange')
          if bound_mn_y is not None:
            axs_dict['mns'][i][j].set_ylim(-bound_mn_y, bound_mn_y)
          

          axs_dict['res_sds'][i][j].scatter(x_plot, resids)
          axs_dict['res_sds'][i][j].plot(cts_x_plot, cts_sds, c='tab:orange')
          
          if plot_loss:
            axs_dict['loss'][i][j].plot(loss_grid[:, i, j])

          if bound_sd_y is not None:
            axs_dict['res_sds'][i][j].set_ylim(-.1, bound_sd_y)
          
      print(i)
      

  fig_res_sds.suptitle('Synthetic: Pred SDs over Residuals ' + str(iteration), size=50)
  fig_res_sds.subplots_adjust(top=0.95)


  fig_mn.suptitle('Synthetic: Means ' + str(iteration), size=50)
  fig_mn.subplots_adjust(top=0.95)

  if plot_loss:
    fig_loss.suptitle('losses ' + str(iteration), size=50)
    fig_loss.subplots_adjust(top=0.95)
  
  if path is not None: 
    fig_mn.savefig(path +'/plots/mean_' + str(iteration) + '.png')
    fig_res_sds.savefig(path +'/plots/res_sd_' + str(iteration) + '.png')
    if plot_loss:
      fig_loss.savefig(path +'/plots/loss_' + str(iteration) + '.png')
  
  if show_plots:
    plt.show()

  plt.close('all')

def num_grad(x, gw):  #n = (1/280) * torch.roll(x, -4) + (-4/105) * torch.roll(x, -3) + (1/5) * torch.roll(x, -2) + (-4/5) * torch.roll(x, -1) + (4/5) * torch.roll(x, 1) + (-1/5) * torch.roll(x, 2) + (4/105) * torch.roll(x, 3) + (-1/280) * torch.roll(x, 4)
  n = (-1/2) * torch.roll(x, -1) + (1/2) * torch.roll(x, 1) 

  d = gw

  return -n / d

def vec_num_grad(x, gw):
  n = (-1/2) * torch.roll(x, -1, 0) + (1/2) * torch.roll(x, 1, 0) 

  d = gw

  return -n / d


def make_heatmap(title, pd_df, xtick, ytick, xlab, ylab, save_path, save=True, symlognorm=True, figsize=(4, 3)):
  # plot figures
  
  plt.figure(figsize = figsize)
  plt.title(title)
  if symlognorm:
    vmin = min(pd_df.min(), 0.)
    vmax = max(pd_df.max(), 1.) 
    norm = SymLogNorm(linthresh=0.03, linscale=0.03,vmin=vmin, vmax=vmax, base=10)
   
    n_ticks = 3
  else:
    norm = None
    n_ticks = 10
    vmin = min(pd_df.min(), 0.)
    vmax = max(pd_df.max(), 1.)


  sns.heatmap(pd_df, annot=False, xticklabels=xtick, yticklabels=ytick, norm=norm, vmin=vmin, vmax=vmax, cbar_kws={'ticks':MaxNLocator(n_ticks), 'format':'%.e'})

  plt.xlabel(xlab)
  plt.ylabel(ylab)
  if save:
    plt.savefig(save_path, dpi=300)

  plt.show()
  plt.close()