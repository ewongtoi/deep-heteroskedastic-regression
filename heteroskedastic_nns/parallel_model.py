import torch
import torch.nn as nn


ACT_FUNCS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "softplus": nn.Softplus,
    "id": nn.Identity,
    "exp": lambda: lambda x: torch.exp(x),
}

def vec_num_grad(x, gw):

  n = (-1/2) * torch.roll(x, -1, 0) + (1/2) * torch.roll(x, 1, 0) 

  d = gw

  return -n / d

def num_grad(x, gw):
  n = (-1/2) * torch.roll(x, -1) + (1/2) * torch.roll(x, 1) 

  d = gw

  return -n / d

class ParallelFF(nn.Module):
    def __init__(
        self, 
        input_size, 
        output_size, 
        hidden_sizes=[100, 100], 
        activation_func="leakyrelu",
        gammas=[1.],
        rhos=[1.],
        init_scale=1e-3,
        precision_activation_func="softplus", 
        per_param_loss=True,
        var_param=False,
        diag=False
    ):
        super().__init__()
        
        activation_func = activation_func.replace(" ", "").lower()
        assert(activation_func in ACT_FUNCS)
        self.activation_func = ACT_FUNCS[activation_func]()

        precision_activation_func = precision_activation_func.replace(" ", "").lower()
        assert(precision_activation_func in ACT_FUNCS)
        self.exp_precision = precision_activation_func == "exp"
        self.precision_activation_func = ACT_FUNCS[precision_activation_func]()

        if not diag:
            # standard start
            hyper_combos = []
            for gamma in gammas:
                for rho in rhos:
                    hyper_combos.append((gamma, rho))
            self.unique_gammas, self.unique_rhos = gammas, rhos
            gammas, rhos = zip(*hyper_combos)
            self.register_buffer("gammas", torch.tensor(gammas, dtype=torch.float))
            self.register_buffer("rhos", torch.tensor(rhos, dtype=torch.float))
            num_models = len(hyper_combos)
            # standard end
        else:
            # diagonal slice start
            self.unique_gammas, self.unique_rhos = gammas, rhos
            
            rev_gamma = [i for i in reversed(gammas)]
            rhos_rep = len(rhos) * rev_gamma
            gammas_rep =[ g for g in gammas for _ in range(len(rhos))]
            rhos_rep = [1-gr for gr in gammas_rep]

            rhos = rhos_rep
            gammas = gammas_rep

            self.register_buffer("gammas", torch.tensor(gammas, dtype=torch.float))
            self.register_buffer("rhos", torch.tensor(rhos, dtype=torch.float))
            num_models = len(gammas)
            # diagonal slice end
        
        
        
        hidden_sizes = [input_size] + hidden_sizes + [output_size]
        layer_sizes = list(zip(hidden_sizes[:-1], hidden_sizes[1:]))
        self.num_layers = len(layer_sizes)

        mean_weights, mean_biases = [], []
        prec_weights, prec_biases = [], []
        for (d_in, d_out) in layer_sizes:
            mean_weights.append(nn.Parameter(torch.randn(num_models, d_in, d_out)*init_scale))
            prec_weights.append(nn.Parameter(torch.randn(num_models, d_in, d_out)*init_scale))
            mean_biases.append(nn.Parameter(torch.randn(num_models, d_out)*init_scale))
            prec_biases.append(nn.Parameter(torch.randn(num_models, d_out)*init_scale))

        self.mean_weights = nn.ParameterList(mean_weights)
        self.mean_biases = nn.ParameterList(mean_biases)
        self.prec_weights = nn.ParameterList(prec_weights)
        self.prec_biases = nn.ParameterList(prec_biases)
        self.num_models = num_models

        mean_param_count = 0
        prec_param_count = 0
        for p in self.mean_weights:
            mean_param_count += p.numel()
        for p in self.mean_biases:
            mean_param_count += p.numel()

        for p in self.prec_weights:
            prec_param_count += p.numel()
        for p in self.prec_biases:
            prec_param_count += p.numel()

        self.mean_param_count = mean_param_count
        self.prec_param_count = prec_param_count
        self.per_param_loss = per_param_loss

        self.var_param=var_param



    def forward(self, x):        
        assert(x.dim() == 2)
        init_form, form = "bi,mio->bmo", "bmi,mio->bmo"

        mean, prec = torch.einsum(init_form, x, self.mean_weights[0]), torch.einsum(init_form, x, self.prec_weights[0])
        mean, prec = mean + self.mean_biases[0], prec + self.prec_biases[0]
            
        for i in range(1, self.num_layers):
            mean, prec = self.activation_func(mean), self.activation_func(prec)
            mean, prec = torch.einsum(form, mean, self.mean_weights[i]), torch.einsum(form, prec, self.prec_weights[i])

            mean, prec = mean + self.mean_biases[i], prec + self.prec_biases[i]

        if not self.get_var_param():
            return {
                "mean": mean,
                "precision_pre_act": prec,
                "precision": (self.precision_activation_func(prec) *  (1 - 1e-8)) + 1e-8
            }
        else:
            return {
                "mean": mean,
                "precision_pre_act": prec,
                "precision": ((self.precision_activation_func(prec) *  (1 - 1e-8)) + 1e-8).pow(-1)
            }

    def get_var_param(self):
        return self.var_param

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
    
    def mean_gam_rho_loss(self, y, pred_results):
        mean, precision = pred_results["mean"], torch.ones_like(pred_results["precision"])

        log_precision = precision.log()


        residuals = (y.unsqueeze(-2) - mean)
        mse = (y.unsqueeze(-2) - mean).pow(2)
        w_mse = mse

        

        all_likelihoods = (w_mse - log_precision).sum(-1)
        model_likelihoods = all_likelihoods.mean(0)

        log_precision = log_precision.sum((0,-1))
        
        if self.per_param_loss:
            raw_mean_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.mean_weights) / self.mean_param_count
            raw_prec_reg = 0
        else:
            raw_mean_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.mean_weights)
            raw_prec_reg = 0

        scaled_mean_reg = self.gammas * raw_mean_reg
        scaled_prec_reg = 0
        total_reg = (1-self.rhos) * (scaled_mean_reg + scaled_prec_reg)
        

        scaled_likelihoods = self.rhos * model_likelihoods
        

        losses = scaled_likelihoods + total_reg
        loss = losses.sum(0)  # sum over model dimension

        safe_losses = torch.where(torch.isnan(losses) | torch.isinf(losses), raw_mean_reg + raw_prec_reg, losses)
        safe_loss = safe_losses.sum(0)
        

        return {
            "losses": losses,
            "loss": loss,
            "safe_losses": safe_losses,
            "safe_loss": safe_loss, 
            "mse": mse,
            "weighted_mse": w_mse,
            "log_precision": log_precision,
            "scaled_mean_reg": scaled_mean_reg,
            "raw_mean_reg": raw_mean_reg,
            "scaled_prec_reg": scaled_prec_reg,
            "raw_prec_reg": raw_prec_reg,
            "residuals": residuals,
            "likelihoods" : model_likelihoods
        }
    
    def mean_gam_rho_loss_laplace(self, y, pred_results):
        mean, precision = pred_results["mean"], torch.ones_like(pred_results["precision"])

        log_precision = precision.log()


        residuals = (y.unsqueeze(-2) - mean)
        mae = (y.unsqueeze(-2) - mean).abs()
        w_mae = mae

    
        all_likelihoods = (w_mae - log_precision).sum(-1)
        model_likelihoods = all_likelihoods.mean(0)

        log_precision = log_precision.sum((0,-1))
        
        if self.per_param_loss:
            raw_mean_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.mean_weights) / self.mean_param_count
            raw_prec_reg = 0
        else:
            raw_mean_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.mean_weights)
            raw_prec_reg = 0

        scaled_mean_reg = self.gammas * raw_mean_reg
        scaled_prec_reg = 0
        total_reg = (1-self.rhos) * (scaled_mean_reg + scaled_prec_reg)
        

        scaled_likelihoods = self.rhos * model_likelihoods
        

        losses = scaled_likelihoods + total_reg
        loss = losses.sum(0)  # sum over model dimension

        safe_losses = torch.where(torch.isnan(losses) | torch.isinf(losses), raw_mean_reg + raw_prec_reg, losses)
        safe_loss = safe_losses.sum(0)
        

        return {
            "losses": losses,
            "loss": loss,
            "safe_losses": safe_losses,
            "safe_loss": safe_loss, 
            "mae": mae,
            "weighted_mae": w_mae,
            "log_precision": log_precision,
            "scaled_mean_reg": scaled_mean_reg,
            "raw_mean_reg": raw_mean_reg,
            "scaled_prec_reg": scaled_prec_reg,
            "raw_prec_reg": raw_prec_reg,
            "residuals": residuals,
            "likelihoods" : model_likelihoods
        }
    
    def beta_nll_loss(self, y, pred_results, beta=0.5):

        mean, precision = pred_results["mean"], pred_results["precision"]


        log_precision = precision.log()

        mse = (y.unsqueeze(-2) - mean).pow(2)
        w_mse = (precision * mse)

        

        all_likelihoods = ((w_mse - log_precision) * precision.pow(-1).detach() ** beta).sum(-1)
        model_likelihoods = all_likelihoods.mean(0)

        log_precision = log_precision.sum((0,-1))
        


        scaled_mean_reg = 0
        scaled_prec_reg = 0
        

        scaled_likelihoods = model_likelihoods
        

        losses = scaled_likelihoods 
        loss = losses.sum(0)  # sum over model dimension

        safe_losses = 0
        safe_loss = 0
        

        return {
            "losses": losses,
            "loss": loss,
            "safe_losses": safe_losses,
            "safe_loss": safe_loss, 
            "mse": mse,
            "weighted_mse": w_mse,
            "log_precision": log_precision,
            "scaled_mean_reg": scaled_mean_reg,
            "raw_mean_reg": 0,
            "scaled_prec_reg": scaled_prec_reg,
            "raw_prec_reg": 0,
            "likelihoods" : model_likelihoods
        }

    
    def gam_rho_loss(self, y, pred_results):
        mean, precision = pred_results["mean"], pred_results["precision"]

        log_precision = precision.log()


        residuals = (y.unsqueeze(-2) - mean)
        mse = (y.unsqueeze(-2) - mean).pow(2)
        w_mse = (precision * mse)

        

        all_likelihoods = (w_mse - log_precision).sum(-1)
        model_likelihoods = all_likelihoods.mean(0)

        log_precision = log_precision.sum((0,-1))
        
        if self.per_param_loss:
            raw_mean_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.mean_weights) / self.mean_param_count
            raw_prec_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.prec_weights) / self.prec_param_count
        else:
            raw_mean_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.mean_weights)
            raw_prec_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.prec_weights)

        scaled_mean_reg = self.gammas * raw_mean_reg
        scaled_prec_reg = (1-self.gammas) * raw_prec_reg
        total_reg = (1-self.rhos) * (scaled_mean_reg + scaled_prec_reg)
        

        scaled_likelihoods = self.rhos * model_likelihoods
        

        losses = scaled_likelihoods + total_reg
        loss = losses.sum(0)  # sum over model dimension

        safe_losses = torch.where(torch.isnan(losses) | torch.isinf(losses), raw_mean_reg + raw_prec_reg, losses)
        safe_loss = safe_losses.sum(0)
        

        return {
            "losses": losses,
            "loss": loss,
            "safe_losses": safe_losses,
            "safe_loss": safe_loss, 
            "mse": mse,
            "weighted_mse": w_mse,
            "log_precision": log_precision,
            "scaled_mean_reg": scaled_mean_reg,
            "raw_mean_reg": raw_mean_reg,
            "scaled_prec_reg": scaled_prec_reg,
            "raw_prec_reg": raw_prec_reg,
            "residuals": residuals,
            "likelihoods" : model_likelihoods
        }
    
    def gam_rho_const_noise_loss(self, y, pred_results):
        mean, precision = pred_results["mean"], pred_results["precision"]
        noise_var = 1./25.

        log_precision = -(precision.pow(-1) + noise_var).log()


        residuals = (y.unsqueeze(-2) - mean)
        mse = (y.unsqueeze(-2) - mean).pow(2) + noise_var
        w_mse = ((precision.pow(-1) + noise_var).pow(-1) * mse)

        

        all_likelihoods = (w_mse - log_precision).sum(-1)
        model_likelihoods = all_likelihoods.mean(0)

        log_precision = log_precision.sum((0,-1))
        
        if self.per_param_loss:
            raw_mean_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.mean_weights) / self.mean_param_count
            raw_prec_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.prec_weights) / self.prec_param_count
        else:
            raw_mean_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.mean_weights)
            raw_prec_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.prec_weights)

        scaled_mean_reg = self.gammas * raw_mean_reg
        scaled_prec_reg = (1-self.gammas) * raw_prec_reg
        total_reg = (1-self.rhos) * (scaled_mean_reg + scaled_prec_reg)
        

        scaled_likelihoods = self.rhos * model_likelihoods
        

        losses = scaled_likelihoods + total_reg
        loss = losses.sum(0)  # sum over model dimension

        safe_losses = torch.where(torch.isnan(losses) | torch.isinf(losses), raw_mean_reg + raw_prec_reg, losses)
        safe_loss = safe_losses.sum(0)
        

        return {
            "losses": losses,
            "loss": loss,
            "safe_losses": safe_losses,
            "safe_loss": safe_loss, 
            "mse": mse,
            "weighted_mse": w_mse,
            "log_precision": log_precision,
            "scaled_mean_reg": scaled_mean_reg,
            "raw_mean_reg": raw_mean_reg,
            "scaled_prec_reg": scaled_prec_reg,
            "raw_prec_reg": raw_prec_reg,
            "residuals": residuals,
            "likelihoods" : model_likelihoods
        }
    
    def gam_rho_loss_laplace(self, y, pred_results):
        mean, precision = pred_results["mean"], pred_results["precision"]

        log_precision = precision.log()


        residuals = (y.unsqueeze(-2) - mean)
        mae = (y.unsqueeze(-2) - mean).abs()
        w_mae = (precision * mae)

        

        all_likelihoods = (w_mae - log_precision).sum(-1)
        model_likelihoods = all_likelihoods.mean(0)

        log_precision = log_precision.sum((0,-1))
        
        if self.per_param_loss:
            raw_mean_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.mean_weights) / self.mean_param_count
            raw_prec_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.prec_weights) / self.prec_param_count
        else:
            raw_mean_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.mean_weights)
            raw_prec_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.prec_weights)

        scaled_mean_reg = self.gammas * raw_mean_reg
        scaled_prec_reg = (1-self.gammas) * raw_prec_reg
        total_reg = (1-self.rhos) * (scaled_mean_reg + scaled_prec_reg)
        

        scaled_likelihoods = self.rhos * model_likelihoods
        

        losses = scaled_likelihoods + total_reg
        loss = losses.sum(0)  # sum over model dimension

        safe_losses = torch.where(torch.isnan(losses) | torch.isinf(losses), raw_mean_reg + raw_prec_reg, losses)
        safe_loss = safe_losses.sum(0)
        

        return {
            "losses": losses,
            "loss": loss,
            "safe_losses": safe_losses,
            "safe_loss": safe_loss, 
            "mae": mae,
            "weighted_mae": w_mae,
            "log_precision": log_precision,
            "scaled_mean_reg": scaled_mean_reg,
            "raw_mean_reg": raw_mean_reg,
            "scaled_prec_reg": scaled_prec_reg,
            "raw_prec_reg": raw_prec_reg,
            "residuals": residuals,
            "likelihoods" : model_likelihoods
        }
    
    def naive_prediction(self, x):


        raw = self(x)

        # output shape is n datapoints, n models, n dim
        mu_bar = raw["mean"].mean(dim=1)
        inv_var_bar = raw["precision"].pow(-1).mean(dim=1).pow(-1)

        return {
            "mean": mu_bar,
            "precision": inv_var_bar
        }

    def mixture_prediction(self, x):

        prec = 0

        raw = self(x)

        raw_means = raw["mean"]

        raw_vars = raw["precision"].pow(-1)

        # output shape is n datapoints, n models, n dim
        
        mu_bar = raw_means.mean(dim=1)
        
        mean_sq_bar = raw_means.pow(2).mean(dim=1)

        var_bar = raw_vars.mean(dim=1)

        prec = (var_bar + mean_sq_bar - mu_bar.pow(2)).pow(-1)


        return {
            "mean": mu_bar,
            "precision": prec
        }
    
    
    
    def grad_pen(self, reg_grid):
        preds = self(reg_grid[:, None])
        gw = reg_grid[1]-reg_grid[0]

        mean_pen = torch.trapezoid(vec_num_grad(preds['mean'], gw).pow(2)[1:-1], reg_grid[1:-1].flatten(), dim=0)
        prec_pen = torch.trapezoid(vec_num_grad(preds['precision'], gw).pow(2)[1:-1], reg_grid[1:-1].flatten(), dim=0)
        
        
        return{"mean_pen": mean_pen, "prec_pen": prec_pen}
    
    def approx_2_int(self, num_pts, eps):

        pt_vec = 2 * (torch.rand((num_pts, 1), device=next(self.parameters()).device) - .5)

        x_pre = pt_vec - eps
        x_post = pt_vec + eps
        
        pred_pre = self(x_pre)
        pred_post = self(x_post)
        
        mgrads = (pred_post['mean'] - pred_pre['mean']) / (2 * eps)

        pgrads = (pred_post['precision'] - pred_pre['precision']) / (2 * eps)

        mints = mgrads.pow(2).mean(0).flatten()
        pints = pgrads.pow(2).mean(0).flatten()

      
        return({'mint': mints, 'pint': pints})


