import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def vec_num_grad(x, gw):
  
    n = (-1/2) * torch.roll(x, -1, 0) + (1/2) * torch.roll(x, 1, 0) 

    d = gw

    return -n / d

def vec_num_grad_uneven(x, gws):
 
    n = (-1/2) * torch.roll(x, -1, 0) + (1/2) * torch.roll(x, 1, 0) 

    d = gws

    return -n / d

def num_grad(x, gw):
 
    n = (-1/2) * torch.roll(x, -1) + (1/2) * torch.roll(x, 1) 

    d = gw

    return -n / d

def vec_linear_interpolation(x_obs, y_obs, x_inter):
    
    x_obs = x_obs.flatten()
    x_inter = x_inter.flatten()
    
    # Sort the observed data by x values
    sorted_indices = torch.argsort(x_obs)
    x_obs_sorted = x_obs[sorted_indices]
    y_obs_sorted = y_obs[:,sorted_indices]

    # Find the indices of the left and right neighboring points for each x_inter
    left_indices = torch.searchsorted(x_obs_sorted.flatten(), x_inter.flatten(), right=True)-1
    right_indices = left_indices + 1
    
 


    # extrapolation above/below
    extrap_a = torch.nonzero((left_indices >= len(x_obs)-1)*1).flatten()
    extrap_b = torch.nonzero((left_indices < 0)*1).flatten()
    
    
    # clamp the problematic values (these will be overwritten anyway)
    right_indices = torch.clamp(right_indices, 0, len(x_obs_sorted) - 1)
    left_indices = torch.clamp(left_indices, 0, len(x_obs_sorted) - 1)

    
    # Get the x values of the left and right neighboring points
    x_left = x_obs_sorted[left_indices]
    x_right = x_obs_sorted[right_indices]

    x_left[extrap_a] = x_obs_sorted[-2]
    x_right[extrap_a] = x_obs_sorted[-1]
    
    x_left[extrap_b] = x_obs_sorted[0]
    x_right[extrap_b] = x_obs_sorted[1]
    

    # Get the y values of the left and right neighboring points
    y_left = y_obs_sorted[:,left_indices]
    y_right = y_obs_sorted[:,right_indices]
  

    y_left[:,extrap_a] = y_obs_sorted[:,[-2]]
    y_right[:,extrap_a] = y_obs_sorted[:,[-1]]
    
    y_left[:,extrap_b] = y_obs_sorted[:,[0]]
    y_right[:,extrap_b] = y_obs_sorted[:,[1]]


    
    # Perform linear interpolation
    slope = (y_right - y_left) / (x_right - x_left).unsqueeze(0).unsqueeze(2)
    y_inter = y_left + slope * (x_inter - x_left).unsqueeze(0).unsqueeze(2)
    
    
    return y_inter


def linear_interpolation(x_obs, y_obs, x_inter):
    # Sort the observed data by x values
    sorted_indices = torch.argsort(x_obs)
    x_obs_sorted = x_obs[sorted_indices]
    y_obs_sorted = y_obs[sorted_indices]

    # Find the indices of the left and right neighboring points for each x_inter
    left_indices = torch.searchsorted(x_obs_sorted, x_inter, right=True)-1
    right_indices = left_indices + 1

    # extrapolation above/below
    extrap_a = torch.nonzero((left_indices >= len(x_obs)-1)*1)
    extrap_b = torch.nonzero((left_indices < 0)*1)
    
    # clamp the problematic values (these will be overwritten anyway)
    right_indices = torch.clamp(right_indices, 0, len(x_obs_sorted) - 1)
    left_indices = torch.clamp(left_indices, 0, len(x_obs_sorted) - 1)

    
    # Get the x values of the left and right neighboring points
    x_left = x_obs_sorted[left_indices]
    x_right = x_obs_sorted[right_indices]

    x_left[extrap_a] = x_obs_sorted[-2]
    x_right[extrap_a] = x_obs_sorted[-1]
    
    x_left[extrap_b] = x_obs_sorted[0]
    x_right[extrap_b] = x_obs_sorted[1]
    

    
    # Get the y values of the left and right neighboring points
    y_left = y_obs_sorted[left_indices]
    y_right = y_obs_sorted[right_indices]
    
    y_left[extrap_a] = y_obs_sorted[-2]
    y_right[extrap_a] = y_obs_sorted[-1]
    
    y_left[extrap_b] = y_obs_sorted[0]
    y_right[extrap_b] = y_obs_sorted[1]

    # Perform linear interpolation
    slope = (y_right - y_left) / (x_right - x_left)
    y_inter = y_left + slope * (x_inter - x_left)

    return y_inter


class ParallelDFT(nn.Module):
    def __init__(
        self,  
        grid_discretization,
        btw_pts=None,
        gammas=[.1, .2], #alpha
        rhos=[.1], # beta
        init_scale=1e-3,
        init_loc=0.,
        split_train=False,
        split_ratio=None
    ):
        super().__init__()

        self.split_train = split_train
        
        hyper_combos = []
        for gamma in gammas:
            for rho in rhos:
                hyper_combos.append((gamma, rho))
        self.unique_gammas, self.unique_rhos = gammas, rhos
        gammas, rhos = zip(*hyper_combos)

        num_models = len(hyper_combos)
        self.register_buffer("gammas", torch.tensor(gammas))
        self.register_buffer("rhos", torch.tensor(rhos))

        num_models = len(gammas)

        gd_min = grid_discretization.min()
        gd_max = grid_discretization.max()
        n_pts = grid_discretization.shape[0]

        self.btw_pts = btw_pts
        gw = grid_discretization[1] - grid_discretization[0]

        if btw_pts is not None:
            aug_n = (n_pts+1) * (btw_pts+1) + 1

            aug_grid_discretization = torch.linspace((gd_min-self.gw).item(), (gd_max+self.gw).item(), aug_n)
        else:
            aug_grid_discretization = grid_discretization

        aug_gw = aug_grid_discretization[1] - aug_grid_discretization[0]

        self.mu_stack = torch.nn.Parameter(torch.randn(aug_grid_discretization.shape[0], num_models, 1)*init_scale + init_loc, requires_grad=True)
        self.log_lam_stack = torch.nn.Parameter(torch.randn(aug_grid_discretization.shape[0], num_models, 1)*init_scale + init_loc, requires_grad=True)

        self.num_models = num_models
        self.register_buffer("grid_discretization", grid_discretization)
        self.register_buffer("aug_grid_discretization", aug_grid_discretization)
        self.register_buffer("aug_gw", aug_gw)
        self.register_buffer("gw", gw)

        if split_train:
            inds = torch.randperm(len(grid_discretization))

            cut_point = int(len(grid_discretization) * split_ratio)
            self.mean_inds = inds[cut_point:]
            self.prec_inds = inds[:cut_point]

            self.merged_mu = self.mu_stack.clone().detach()
            self.merged_log_lam = self.log_lam_stack.clone().detach()




    def full_integral_loss(self, stoch_y):
        assert self.btw_pts is None, "interpolation"
        precision = self.log_lam_stack.exp()
        mu = self.mu_stack
        res = (stoch_y.unsqueeze(-2) - mu)


        likelihood = (res.pow(2) * precision - self.log_lam_stack).mean(0)
        raw_mean_pens = torch.trapezoid(vec_num_grad(mu, self.gw).pow(2)[1:-1], self.grid_discretization[1:-1].flatten(), dim=0)
        raw_prec_pens = torch.trapezoid(vec_num_grad(precision, self.gw).pow(2)[1:-1], self.grid_discretization[1:-1].flatten(), dim=0)

        mean_pens = self.gammas.view(self.num_models, 1) * raw_mean_pens
        prec_pens = self.rhos.view(self.num_models, 1) * raw_prec_pens


        return {'agg': likelihood.mean() + mean_pens.mean() + prec_pens.mean(),
              'likelihood': likelihood,
              'mean_pens': mean_pens,
              'raw_mean_pens': raw_mean_pens,
              'prec_pens': prec_pens,
              'raw_prec_pens': raw_prec_pens}

    def gamma_rho_integral_loss(self, stoch_y):
        assert self.btw_pts is None, "interpolation"

        # data pts x num models x 1
        precision = self.log_lam_stack.exp()
        mu = self.mu_stack

        # stoch_y is data points x 1
        resids = (stoch_y.unsqueeze(-2) - mu)

        # data pts x num modelx x 1
        mse = resids.pow(2)

        # data pts x num_models x 1
        w_mse = (precision * mse) 

        # num models
        log_precision = self.log_lam_stack.sum((0,-1))

        # data points x num models
        all_likelihoods = (w_mse - self.log_lam_stack).sum(-1)

        # num models
        model_likelihoods = all_likelihoods.mean(0)

        raw_mean_pens = torch.trapezoid(vec_num_grad(mu, self.gw).pow(2)[1:-1], self.grid_discretization[1:-1].flatten(), dim=0)
        raw_prec_pens = torch.trapezoid(vec_num_grad(precision, self.gw).pow(2)[1:-1], self.grid_discretization[1:-1].flatten(), dim=0)


        scaled_likelihoods = self.rhos * model_likelihoods


        mean_pens = self.gammas * raw_mean_pens.flatten()
        prec_pens = (1-self.gammas) * raw_prec_pens.flatten()
        total_pen = (1-self.rhos) * (mean_pens + prec_pens)

        losses = scaled_likelihoods + total_pen


        return {'loss': losses.sum(),
              'losses': losses,
              'mean_pens': mean_pens,
              'raw_mean_pens': raw_mean_pens,
              'prec_pens': prec_pens,
              'raw_prec_pens': raw_prec_pens,
              'resids': resids,
              'log_precision': log_precision,
              'wmse': w_mse.sum((0, -1))}

    def gamma_rho_split_loss(self, stoch_y):
        assert self.btw_pts is None, "interpolation"

        
        interpolated_mean = vec_linear_interpolation(self.grid_discretization[self.mean_inds], self.mu_stack[self.mean_inds,:,:].reshape(self.num_models, len(self.mean_inds),  1), self.grid_discretization[self.prec_inds]).detach()
        
        interpolated_log_lam = vec_linear_interpolation(self.grid_discretization[self.prec_inds], self.log_lam_stack[self.prec_inds,:,:].reshape(self.num_models, len(self.prec_inds), 1), self.grid_discretization[self.mean_inds]).detach()
      
        
        

        # data pts x num models x 1
        mu = self.mu_stack

        # replace the frozen values with the interpolated ones
        merged_mu = mu.detach().clone()
        merged_log_lam = self.log_lam_stack.detach().clone()

        # indices to change
        merged_mu[self.mean_inds, :, 0] = mu[self.mean_inds, :, 0] 
        merged_mu[self.prec_inds, :, 0] = interpolated_mean.reshape(len(self.prec_inds), self.num_models)
        merged_log_lam[self.prec_inds, :, 0] = self.log_lam_stack[self.prec_inds, :, 0]
        merged_log_lam[self.mean_inds, :, 0] = interpolated_log_lam.reshape(len(self.mean_inds), self.num_models)


        # stoch_y is data points x 1
        resids = (stoch_y.unsqueeze(-2) - merged_mu)


        # data pts x num modelx x 1
        mse = resids.pow(2)

        # data pts x num_models x 1
        w_mse = (merged_log_lam.exp() * mse) 

        # num models
        log_precision = merged_log_lam.sum((0,-1))

        # data points x num models
        all_likelihoods = (w_mse - merged_log_lam).sum(-1)

        # num models
        model_likelihoods = all_likelihoods.mean(0)

        raw_mean_pens = torch.trapezoid(vec_num_grad(merged_mu, self.gw).pow(2)[1:-1], self.grid_discretization[1:-1].flatten(), dim=0)
        raw_prec_pens = torch.trapezoid(vec_num_grad(merged_log_lam.exp(), self.gw).pow(2)[1:-1], self.grid_discretization[1:-1].flatten(), dim=0)


        scaled_likelihoods = self.rhos * model_likelihoods


        mean_pens = self.gammas * raw_mean_pens.flatten()
        prec_pens = (1-self.gammas) * raw_prec_pens.flatten()
        total_pen = (1-self.rhos) * (mean_pens + prec_pens)

        losses = scaled_likelihoods + total_pen

        self.merged_mu = merged_mu
        self.merged_log_lam = merged_log_lam


        return {'loss': losses.sum(),
              'losses': losses,
              'mean_pens': mean_pens,
              'raw_mean_pens': raw_mean_pens,
              'prec_pens': prec_pens,
              'raw_prec_pens': raw_prec_pens,
              'resids': resids,
              'log_precision': log_precision,
              'wmse': w_mse.sum((0, -1))}


    def interpolation_integral_loss(self, stoch_y):
        assert self.btw_pts is not None, "no interpolation"

        precision = self.log_lam_stack.exp()
        mu = self.mu_stack

        inter_gap = self.btw_pts + 1

        res = (stoch_y.unsqueeze(-2) - mu[inter_gap:self.aug_grid_discretization.shape[0]-inter_gap:inter_gap, :, :])

        likelihood = (res.pow(2) * precision[inter_gap:self.aug_grid_discretization.shape[0]-inter_gap:inter_gap, :, :] - self.log_lam_stack[inter_gap:self.aug_grid_discretization.shape[0]-inter_gap:inter_gap, :, :]).mean(0)

      
        raw_mean_pens = torch.trapezoid(vec_num_grad(mu, self.aug_gw).pow(2)[1:-1], self.aug_grid_discretization[1:-1].flatten(), dim=0)
        mean_pens = self.gammas.view(self.num_models, 1) * raw_mean_pens

        raw_prec_pens = torch.trapezoid(vec_num_grad(precision, self.aug_gw).pow(2)[1:-1], self.aug_grid_discretization[1:-1].flatten(), dim=0)
        prec_pens = self.rhos.view(self.num_models, 1) * raw_prec_pens


        return {'agg': likelihood.mean() + mean_pens.mean() + prec_pens.mean(),
              'likelihood': likelihood,
              'mean_pens': mean_pens,
              'raw_prec_pens': raw_mean_pens,
              'prec_pens': prec_pens,
              'raw_prec_pens': raw_prec_pens}


    def loss(self, y, pred_results):
        mean, precision = pred_results["mean"], pred_results["precision"]
        # if self.exp_precision:
        #    log_precision = pred_results["precision_pre_act"]
        # else:
        log_precision = precision.log()

        residuals = (y.unsqueeze(-2) - mean)
        mse = (y.unsqueeze(-2) - mean).pow(2)
        w_mse = (precision * mse).sum((0, -1))  # sum over batch dimension and output dimension
        mse = mse.sum((0, -1))
        log_precision = log_precision.sum((0,-1))

        raw_mean_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.mean_weights)
        raw_prec_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.prec_weights)

        mean_reg = self.gammas * raw_mean_reg
        prec_reg = self.rhos * raw_prec_reg

        losses = w_mse - log_precision + mean_reg + prec_reg
        loss = losses.sum(0)  # sum over model dimension

        safe_losses = torch.where(torch.isnan(losses) | torch.isinf(losses), raw_mean_reg + raw_prec_reg, losses)
        safe_loss = safe_losses.sum(0)
        
        likelihood = (w_mse - log_precision).sum(0)

        return {
            "losses": losses,
            "loss": loss,
            "safe_losses": safe_losses,
            "safe_loss": safe_loss, 
            "mse": mse,
            "weighted_mse": w_mse,
            "log_precision": log_precision,
            "mean_reg": mean_reg,
            "raw_mean_reg": raw_mean_reg,
            "prec_reg": prec_reg,
            "raw_prec_reg": raw_prec_reg,
            "residuals": residuals,
            "likelihood" : likelihood
        }

    # plots only data on the actual data points (no interpolation), even if interpolating while training
    def plot(self, y, epochs, plot_loss=None, bound_sd_y=None, bound_mn_y=None, stats=None, save_path=None, show_plots=False):
      
        gammas = self.unique_gammas
        rhos = self.unique_rhos

        plot_loss = stats is not None

        if plot_loss:
            loss_grid = torch.stack([l['losses'].reshape(len(gammas), len(rhos)).cpu().detach() for l in stats])

        a_labs = self.gammas.reshape(len(gammas), len(rhos)).cpu().detach() 
        b_labs = self.rhos.reshape(len(gammas), len(rhos)).cpu().detach() 


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
        if self.btw_pts is None:
            pm_mns = self.merged_mu if self.split_train  else self.mu_stack
            pm_mns = pm_mns.reshape((len(self.grid_discretization), len(gammas), len(rhos))).cpu().detach() 
            pm_sds = self.merged_log_lam.exp().pow(-.5) if self.split_train else self.log_lam_stack.exp().pow(-0.5)
            pm_sds = pm_sds.reshape((len(self.grid_discretization), len(gammas), len(rhos))).cpu().detach() 


            x_plot = self.grid_discretization.detach().cpu()
            y_plot = y.cpu().detach().flatten()

            # each value of reg for the mean network
            for i in range(len(gammas)):

                # each value of reg for the prec/sd network
                for j in range(len(rhos)):


                    if j == 0:
                        for _, axs in axs_dict.items():
                            axs[i][j].set_ylabel(r"$\gamma$: {:.3E}".format(a_labs[i][0]))

                    if i == len(gammas)-1:
                        for _, axs in axs_dict.items():
                            axs[i][j].set_xlabel(r"$\rho$: {:.3E}".format(b_labs[0][j]))


                    mns = pm_mns[:, i, j]
                    sds = pm_sds[:, i, j]

                    resids = (mns - y_plot).abs()

                    if self.split_train:
                        axs_dict['mns'][i][j].scatter(x_plot[self.mean_inds], y_plot[self.mean_inds], marker='+')
                        axs_dict['mns'][i][j].scatter(x_plot[self.prec_inds], y_plot[self.prec_inds], marker='.')
                    else:

                        axs_dict['mns'][i][j].scatter(x_plot, y_plot)
                    
                    axs_dict['mns'][i][j].plot(x_plot, mns, c='orange')


                    axs_dict['mns'][i][j].fill_between(x_plot.flatten(), (mns-sds).flatten(), (mns+sds).flatten(), color='b', alpha=.2)
                    axs_dict['mns'][i][j].fill_between(x_plot.flatten(), (mns-2*sds).flatten(), (mns+2*sds).flatten(), color='b', alpha=.1)

                    if bound_mn_y is not None:
                        axs_dict['mns'][i][j].set_ylim(-bound_mn_y, bound_mn_y)


                    if self.split_train:
                        axs_dict['res_sds'][i][j].scatter(x_plot[self.mean_inds], resids[self.mean_inds], marker='+')
                        axs_dict['res_sds'][i][j].scatter(x_plot[self.prec_inds], resids[self.prec_inds], marker='.')
                    else:
                        axs_dict['res_sds'][i][j].scatter(x_plot, resids)
                    
                    axs_dict['res_sds'][i][j].plot(x_plot, sds, c='orange')

                    if plot_loss:
                        axs_dict['loss'][i][j].plot(loss_grid[:, i, j])

                    if bound_sd_y is not None:
                        axs_dict['res_sds'][i][j].set_ylim(-bound_sd_y, bound_sd_y)



                print(i)



            fig_res_sds.suptitle('Synthetic: Pred SDs over Residuals ' + str(epochs), size=50)
            fig_res_sds.subplots_adjust(top=0.95)


            fig_mn.suptitle('Synthetic: Means ' + str(epochs), size=50)
            fig_mn.subplots_adjust(top=0.95)

            if plot_loss:
                fig_loss.suptitle('losses ' + str(epochs), size=50)
                fig_loss.subplots_adjust(top=0.95)
        else:
            inter_gap = self.btw_pts + 1
            pm_mns = self.mu_stack[inter_gap:self.aug_grid_discretization.shape[0]-inter_gap:inter_gap, :, :]
            pm_mns = pm_mns.reshape((len(self.grid_discretization), len(gammas), len(rhos))).cpu().detach() 
            pm_sds = self.log_lam_stack.exp().pow(-.5)[inter_gap:self.aug_grid_discretization.shape[0]-inter_gap:inter_gap, :, :]
            pm_sds = pm_sds.reshape((len(self.grid_discretization), len(gammas), len(rhos))).cpu().detach() 


            x_plot = self.grid_discretization.detach().cpu()
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

                    resids = (mns - y_plot).abs()

                    axs_dict['mns'][i][j].scatter(x_plot, y_plot)

                    axs_dict['mns'][i][j].fill_between(x_plot.flatten(), (mns-sds).flatten(), (mns+sds).flatten(), color='b', alpha=.2)
                    axs_dict['mns'][i][j].fill_between(x_plot.flatten(), (mns-2*sds).flatten(), (mns+2*sds).flatten(), color='b', alpha=.1)

                    axs_dict['mns'][i][j].plot(x_plot, mns, c='orange')

                    if bound_mn_y is not None:
                        axs_dict['mns'][i][j].set_ylim(-bound_mn_y, bound_mn_y)


                    axs_dict['res_sds'][i][j].scatter(x_plot, resids)
                    axs_dict['res_sds'][i][j].plot(x_plot, sds, c='orange')

                    if plot_loss:
                        axs_dict['loss'][i][j].plot(loss_grid[:, i, j])

                    if bound_sd_y is not None:
                        axs_dict['res_sds'][i][j].set_ylim(-bound_sd_y, bound_sd_y)



                print(i)



            fig_res_sds.suptitle('Synthetic: Pred SDs over Residuals ' + str(epochs), size=50)
            fig_res_sds.subplots_adjust(top=0.95)


            fig_mn.suptitle('Synthetic: Means ' + str(epochs), size=50)
            fig_mn.subplots_adjust(top=0.95)

            if plot_loss:
                fig_loss.suptitle('losses ' + str(epochs), size=50)
                fig_loss.subplots_adjust(top=0.95)

      
        if save_path is not None: 
            fig_mn.savefig(save_path +'/plots/mean_' + str(epochs) + '.pdf')
            fig_res_sds.savefig(save_path +'/plots/res_sd_' + str(epochs) + '.pdf')
        if plot_loss:
            fig_loss.savefig(save_path +'/plots/loss_' + str(epochs) + '.pdf')

        if show_plots:
            plt.show()

        plt.close('all')
    
    # plots interpolated learned functions \mu, \Lambda^{-.5}
    def inter_plot(self, y, epochs, plot_loss=None, bound_sd_y=None, bound_mn_y=None, stats=None):
        assert self.btw_pts is not None, "no interpolation"
        gammas = self.unique_gammas
        rhos = self.unique_rhos

        inter_gap = self.btw_pts + 1

        plot_loss = stats is not None

        if plot_loss:
            loss_grid = torch.stack([l['losses'] for l in stats]).reshape(len(stats), len(gammas), len(rhos)).cpu().detach() 

        a_labs = self.gammas.reshape(len(gammas), len(rhos)).cpu().detach() 
        b_labs = self.rhos.reshape(len(gammas), len(rhos)).cpu().detach() 


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

        pm_mns = self.mu_stack
        pm_mns = pm_mns.reshape((len(self.aug_grid_discretization), len(gammas), len(rhos))).cpu().detach() 
        pm_sds = self.log_lam_stack.exp().pow(-.5)
        pm_sds = pm_sds.reshape((len(self.aug_grid_discretization), len(gammas), len(rhos))).cpu().detach() 


        x_plot = self.grid_discretization.detach().cpu()
        y_plot = y.cpu().detach().flatten()
        inter_gap = self.btw_pts + 1

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

                resids = (mns[inter_gap:self.aug_grid_discretization.shape[0]-inter_gap:inter_gap] - y_plot).abs()

                axs_dict['mns'][i][j].scatter(self.grid_discretization.cpu(), y_plot)
                axs_dict['mns'][i][j].plot(self.aug_grid_discretization.cpu(), mns, c='orange')
              
                if bound_mn_y is not None:
                    axs_dict['mns'][i][j].set_ylim(-bound_mn_y, bound_mn_y)
                  
              
                axs_dict['res_sds'][i][j].scatter(self.grid_discretization.cpu(), resids)
                axs_dict['res_sds'][i][j].plot(self.aug_grid_discretization.cpu(), sds, c='orange')
              
                if plot_loss:
                    axs_dict['loss'][i][j].plot(loss_grid[:, i, j])

                if bound_sd_y is not None:
                    axs_dict['res_sds'][i][j].set_ylim(-bound_sd_y, bound_sd_y)

          

            print(i)
          


        fig_res_sds.suptitle('Synthetic: Pred SDs over Residuals ' + str(epochs), size=50)
        fig_res_sds.subplots_adjust(top=0.95)


        fig_mn.suptitle('Synthetic: Means ' + str(epochs), size=50)
        fig_mn.subplots_adjust(top=0.95)

        if plot_loss:
            fig_loss.suptitle('losses ' + str(epochs), size=50)
            fig_loss.subplots_adjust(top=0.95)
      
