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


class LSFF(nn.Module):
    def __init__(
        self, 
        input_size, 
        output_size, 
        hidden_sizes=[100, 100], 
        activation_func="leakyrelu",
        init_scale=1e-3,
        scale_activation_func="softplus", 
        inv_param=False,
        gammas=None,
        rhos=None,
        diag=False,
        family=None,
        B_sigma=1.,
        B_dim=10
    ):
        super().__init__()



        self.family=family
        activation_func = activation_func.replace(" ", "").lower()
        assert(activation_func in ACT_FUNCS)
        self.activation_func = ACT_FUNCS[activation_func]()

        scale_activation_func = scale_activation_func.replace(" ", "").lower()
        assert(scale_activation_func in ACT_FUNCS)
        self.exp_scale = scale_activation_func == "exp"
        # function
        self.scale_activation_func = ACT_FUNCS[scale_activation_func]()

        # name of function
        self.scale_activation = scale_activation_func

        self.diag = diag
        
        if not diag:
            hyper_combos = []
            for gamma in gammas:
                for rho in rhos:
                    hyper_combos.append((gamma, rho))
            self.unique_gammas, self.unique_rhos = gammas, rhos
            gammas, rhos = zip(*hyper_combos)
            self.register_buffer("gammas", torch.tensor(gammas, dtype=torch.float))
            self.register_buffer("rhos", torch.tensor(rhos, dtype=torch.float))
            num_models = len(hyper_combos)

            
        else:

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

        if B_dim is not None and B_sigma is not None:
            B_mat = torch.randn((num_models, input_size, B_dim)) * B_sigma * torch.pi * 2
            hidden_sizes = [B_dim * 2] + hidden_sizes + [output_size]
            layer_sizes = list(zip(hidden_sizes[:-1], hidden_sizes[1:]))
            self.num_layers = len(layer_sizes)
            
        else:
            hidden_sizes = [input_size] + hidden_sizes + [output_size]
            layer_sizes = list(zip(hidden_sizes[:-1], hidden_sizes[1:]))
            self.num_layers = len(layer_sizes)
            B_mat = None
        
        self.register_buffer("B_mat", B_mat)
        

        


        location_weights, location_biases = [], []
        scale_weights, scale_biases = [], []
        for (d_in, d_out) in layer_sizes:
            location_weights.append(nn.Parameter(torch.randn(num_models, d_in, d_out)*init_scale))
            scale_weights.append(nn.Parameter(torch.randn(num_models, d_in, d_out)*init_scale))
            location_biases.append(nn.Parameter(torch.randn(num_models, d_out)*init_scale))
            scale_biases.append(nn.Parameter(torch.randn(num_models, d_out)*init_scale))

        self.location_weights = nn.ParameterList(location_weights)
        self.location_biases = nn.ParameterList(location_biases)
        self.scale_weights = nn.ParameterList(scale_weights)
        self.scale_biases = nn.ParameterList(scale_biases)
        self.num_models = num_models
        


        location_param_count = 0
        scale_param_count = 0
        for p in self.location_weights:
            location_param_count += p.numel()
        for p in self.location_biases:
            location_param_count += p.numel()

        for p in self.scale_weights:
            scale_param_count += p.numel()
        for p in self.scale_biases:
            scale_param_count += p.numel()

        self.location_param_count = location_param_count

        self.inv_param=inv_param



    def fourier_forward(self, x):        
        assert(x.dim() == 3)
        form = "bmi,mio->bmo"



        transformed_x = torch.einsum(form, x, self.B_mat)
        transformed_x = torch.cat([torch.cos(transformed_x), torch.sin(transformed_x)], dim=-1)

        location, scale = transformed_x, transformed_x
        

        for i in range(self.num_layers):

            if i != 0 :
                location, scale = self.activation_func(location), self.activation_func(scale)

            location, scale = torch.einsum(form, location, self.location_weights[i]), torch.einsum(form, scale, self.scale_weights[i])

            location, scale = location + self.location_biases[i], scale + self.scale_biases[i]

        # model sd
        if (not self.get_inv_param()) and self.family=="gaussian":
            return {
                "location": location,
                "preact_scale": scale,
                "scale": (self.scale_activation_func(scale) *  (1 - 1e-8)) + 1e-8
            }
        # model precision
        elif self.family=="gaussian" and self.get_inv_param():
            return {
                "location": location,
                "precision": (self.scale_activation_func(scale) *  (1 - 1e-8)) + 1e-8,
                "scale": ((self.scale_activation_func(scale) *  (1 - 1e-8)) + 1e-8).pow(-.5) + 1e-8
            }
        else:
            return {
                "location": location,
                "preact_scale": scale,
                "scale": (self.scale_activation_func(scale) *  (1 - 1e-8)) + 1e-8
            }

    def forward(self, x):        
        assert(x.dim() == 3)
        form = "bmi,mio->bmo"

        location, scale = x, x
        

        for i in range(self.num_layers):

            if i != 0 :
                location, scale = self.activation_func(location), self.activation_func(scale)

            location, scale = torch.einsum(form, location, self.location_weights[i]), torch.einsum(form, scale, self.scale_weights[i])

            location, scale = location + self.location_biases[i], scale + self.scale_biases[i]

        # model sd
        if (not self.get_inv_param()) and self.family=="gaussian":
            return {
                "location": location,
                "preact_scale": scale,
                "scale": (self.scale_activation_func(scale) *  (1 - 1e-8)) + 1e-8
            }
        # model precision
        elif self.family=="gaussian" and self.get_inv_param():
            return {
                "location": location,
                "precision": (self.scale_activation_func(scale) *  (1 - 1e-8)) + 1e-8,
                "scale": ((self.scale_activation_func(scale) *  (1 - 1e-8)) + 1e-8).pow(-.5) + 1e-8
            }
        else:
            return {
                "location": location,
                "preact_scale": scale,
                "scale": (self.scale_activation_func(scale) *  (1 - 1e-8)) + 1e-8
            }

    def get_inv_param(self):
        return self.inv_param

    # redirects to correct location only loss for a given likelihood
    def location_loss(self, y, pred_results):
        if self.family == "gaussian":
            return(self.mean_gaussian_loss(y, pred_results))
        elif self.family == "laplace":
            return(self.median_laplace_loss(y, pred_results))
        elif self.family == "cauchy":
            return(self.median_cauchy_loss(y, pred_results))        
        else:
            return("misspecified family")
        

    # loss only for the mean portion of gaussian
    def mean_gaussian_loss(self, y, pred_results):
        assert(self.family == "gaussian" or self.family == "natural")

        mean, precision = pred_results["location"], torch.ones_like(pred_results["scale"])

        log_precision = precision.log()


        mse = (y - mean).pow(2)
        w_mse = mse

        all_likelihoods = (w_mse - log_precision).sum(-1)
        model_likelihoods = all_likelihoods.mean(0)

        log_precision = log_precision.sum((0,-1))

        
        losses = model_likelihoods
        loss = losses.sum(0)  # sum over model dimension


        return {
            "losses": losses,
            "loss": loss
        }

    # loss only for the sd portion of gaussian
    def sd_gaussian_loss(self, y, pred_results):
        assert(self.family == "gaussian")

        if self.get_inv_param():
            mean, precision = torch.zeros_like(pred_results["location"]), pred_results["precision"]
        else:
            mean, precision = torch.zeros_like(pred_results["location"]), pred_results["scale"].pow(-2)

        log_precision = precision.log()


        mse = (y - mean).pow(2)
        w_mse = mse * precision

        all_likelihoods = (w_mse - log_precision).sum(-1)
        model_likelihoods = all_likelihoods.mean(0)

        log_precision = log_precision.sum((0,-1))

        
        losses = model_likelihoods
        loss = losses.sum(0)  # sum over model dimension


        return {
            "losses": losses,
            "loss": loss
        }
    
    
    # loss only on the location portion of the laplace
    def median_laplace_loss(self, y, pred_results):
        assert(self.family == "laplace")

        mean, scale = pred_results["location"], torch.ones_like(pred_results["scale"])

        log_scale = scale.log()


        mae = (y - mean).abs()
        w_mae = mae

        
        all_likelihoods = (w_mae - log_scale).sum(-1)
        model_likelihoods = all_likelihoods.mean(0)

        log_scale = log_scale.sum((0,-1))


        scaled_likelihoods = model_likelihoods
        

        losses = scaled_likelihoods 
        loss = losses.sum(0)  # sum over model dimension

        

        return {
            "losses": losses,
            "loss": loss,
            "mae": mae,
            "weighted_mae": w_mae,
            "log_precision": log_scale,
            "likelihoods" : model_likelihoods
        }
    
    
    # loss only on the location portion of the cauchy
    def median_cauchy_loss(self, y, pred_results):
        assert(self.family == "cauchy")

        mean, scale = pred_results["location"], torch.ones_like(pred_results["scale"])

        log_scale = scale.log()


  
        #mse = (y.unsqueeze(-2) - mean).pow(2)
        mse = (y - mean).pow(2)
        wmse = mse / scale.pow(2)

        

        all_likelihoods = (log_scale + torch.log(1 + wmse)).sum(-1)
        model_likelihoods = all_likelihoods.mean(0)

        log_scale = log_scale.sum((0,-1))

        

        scaled_likelihoods = model_likelihoods
        

        losses = scaled_likelihoods 
        loss = losses.sum(0)  # sum over model dimension

        

        return {
            "losses": losses,
            "loss": loss
        }


    # redirects to correct loss for a given likelihood
    def loss(self, y, pred_results):
        if self.family == "gaussian":
            return(self.gaussian_loss(y, pred_results))
        elif self.family == "laplace":
            return(self.laplace_loss(y, pred_results))
        elif self.family == "cauchy":
            return(self.cauchy_loss(y, pred_results))
        elif self.family == "natural":
            return(self.gaussian_nat_loss(y, pred_results))
        else:
            return("misspecified family")
        

    # full heteroskedastic gaussian loss    
    def gaussian_loss(self, y, pred_results):
        assert(self.family == "gaussian")

        mean, precision = pred_results["location"], pred_results["scale"].pow(-2)

        log_precision = precision.log()

        mse = (y - mean).pow(2)
        w_mse = (precision * mse)

        

        all_likelihoods = (w_mse - log_precision).sum(-1)
        model_likelihoods = all_likelihoods.mean(0)

        log_precision = log_precision.sum((0,-1))
        

        losses = model_likelihoods
        loss = losses.sum(0)  # sum over model dimension
      

        return {
            "losses": losses,
            "loss": loss,
            "mse": mse,
            "weighted_mse": w_mse,
            "log_precision": log_precision,
            "likelihoods" : model_likelihoods
        }
    
    # full heteroskedastic gaussian loss    
    def gaussian_nat_loss(self, y, pred_results):
        assert(self.family == "natural")

        eta_1, eta_2 = pred_results["location"], -pred_results["scale"]
        


        

        all_likelihoods = (eta_1 * y - eta_2 * y.pow(2) - (eta_1.pow(2) / (4 * eta_2)) - .5 * torch.log(-2 * eta_2))
 
        all_likelihoods = all_likelihoods.sum(-1)
        model_likelihoods = all_likelihoods.mean(0)
        

        losses = model_likelihoods
        loss = losses.sum(0)  # sum over model dimension
      

        return {
            "losses": losses,
            "loss": loss,
            "likelihoods" : model_likelihoods
        }
    
 
    def gaussian_gam_rho_de_loss(self, y, pred_results):
        assert(self.family == "gaussian")

        # model the precision directly
        if self.activation_func == "exp":
            if self.get_inv_param():
                mean, precision = pred_results["location"], pred_results["precision"]
                log_precision = pred_results["preact_scale"] 
            else:
                mean, precision = pred_results["location"], pred_results["precision"]
                log_precision = pred_results["preact_scale"] * (-2.)
        else:
            if self.get_inv_param():
                mean, precision = pred_results["location"], pred_results["precision"]
                log_precision = precision.log()
            else:
                # convert scale to precision
                mean, precision = pred_results["location"], pred_results["scale"].pow(-2)
                log_precision = precision.log()
        



        mse = (y - mean).pow(2)
        w_mse = (precision * mse)

        

        all_likelihoods = .5 * (w_mse - log_precision).sum(-1)
        model_likelihoods = all_likelihoods.mean(0)

        log_precision = log_precision.sum((0,-1))

        # square, sum over input dimension, average over datapoints
        mean_de = pred_results['loc_grad'].pow(2).sum(-2).mean(0)
        prec_de = pred_results['scale_grad'].pow(2).sum(-2).mean(0) 


        scaled_mean_reg = self.gammas * mean_de.squeeze(-1)
        scaled_prec_reg = (1-self.gammas) * prec_de.squeeze(-1)
        
        
        total_reg = (1-self.rhos) * (scaled_mean_reg + scaled_prec_reg)
        

        scaled_likelihoods = self.rhos * model_likelihoods
        

        losses = scaled_likelihoods + total_reg
        loss = losses.sum() 

        

        return {
            "losses": losses,
            "loss": loss,
            "model_likelihoods": model_likelihoods,
            "mean_de": mean_de,
            "prec_de": prec_de
        }

    def gaussian_gam_rho_loss(self, y, pred_results):
        # model the precision directly
        if self.get_inv_param():
            mean, precision = pred_results["location"], pred_results["precision"]
        
        else:
            # convert scale to precision
            mean, precision = pred_results["location"], pred_results["scale"].pow(-2)



        log_precision = precision.log()


        residuals = (y - mean)
        mse = (y - mean).pow(2)
        w_mse = (precision * mse)

        

        all_likelihoods = (w_mse - log_precision).sum(-1)
        model_likelihoods = all_likelihoods.mean(0)

        log_precision = log_precision.sum((0,-1))
        

        raw_mean_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.location_weights)
        raw_prec_reg = sum(weight.pow(2).sum((-2, -1)) for weight in self.scale_weights)

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
            "mse": mse,
            "raw_mean_reg": raw_mean_reg,
            "raw_prec_reg": raw_prec_reg,
            "likelihoods" : model_likelihoods
        }

    # full laplace loss
    def laplace_loss(self, y, pred_results):
        assert(self.family=="laplace")

        mean, scale = pred_results["location"], pred_results["scale"]

        log_scale = scale.log()


        
        #mae = (y.unsqueeze(-2) - mean).abs()
        mae = (y - mean).abs()
        w_mae = (scale * mae)

        

        all_likelihoods = (w_mae - log_scale).sum(-1)
        model_likelihoods = all_likelihoods.mean(0)

        log_scale = log_scale.sum((0,-1))
        

        losses = model_likelihoods
        loss = losses.sum(0)  # sum over model dimension


        return {
            "losses": losses,
            "loss": loss,
            "mae": mae,
            "weighted_mae": w_mae,
            "log_scale": log_scale,
            "likelihoods" : model_likelihoods
        }
    
    
    # full cauchy loss
    def cauchy_loss(self, y, pred_results):
        assert(self.family=="cauchy")

        median, scale = pred_results["location"], pred_results["scale"]

        log_scale = scale.log()


        
        #mse = (y.unsqueeze(-2) - median).pow(2)
        mse = (y - median).pow(2)
        wmse = (mse / scale.pow(2))

        

        all_likelihoods = (log_scale + torch.log(1 + wmse)).sum(-1)
        model_likelihoods = all_likelihoods.mean(0)

        log_scale = log_scale.sum((0,-1))
        

        losses = model_likelihoods
        loss = losses.sum(0)  # sum over model dimension


        return {
            "losses": losses,
            "loss": loss
        }
    
    
    # averages means and variances for guassian output
    def naive_prediction(self, x):


        raw = self(x)

        # output shape is n datapoints, n models, n dim
        mu_bar = raw["location"].mean(dim=1)
        inv_var_bar = raw["scale"].pow(-1).mean(dim=1).pow(-1)

        return {
            "location": mu_bar,
            "scale": inv_var_bar.pow(-1)
        }

    # treats the ensemble as a mixture of gaussians
    def mixture_prediction(self, x):
        assert(self.family == "gaussian")

        prec = 0

        raw = self(x)

        raw_means = raw["location"]

        raw_vars = raw["scale"].pow(-1)

        # output shape is n datapoints, n models, n dim
        
        mu_bar = raw_means.mean(dim=1)
        
        mean_sq_bar = raw_means.pow(2).mean(dim=1)

        var_bar = raw_vars.mean(dim=1)

        prec = (var_bar + mean_sq_bar - mu_bar.pow(2)).pow(-1)


        return {
            "location": mu_bar,
            "scale": prec.pow(-1)
        }
    
    # beta nll loss from seitzer (assumes normal)
    def beta_nll_loss(self, y, pred_results, beta=0.5):

        mean, precision = pred_results["location"], pred_results["scale"]


        log_precision = precision.log()


      
        #mse = (y.unsqueeze(-2) - mean).pow(2)
        mse = (y - mean).pow(2)
        w_mse = (precision * mse)

        

        all_likelihoods = ((w_mse - log_precision) * precision.pow(-1).detach() ** beta).sum(-1)
        model_likelihoods = all_likelihoods.mean(0)

        log_precision = log_precision.sum((0,-1))
        

        

        losses = model_likelihoods
        loss = losses.sum(0)  # sum over model dimension

        

        return {
            "losses": losses,
            "loss": loss,
            "mse": mse,
            "weighted_mse": w_mse,
            "log_precision": log_precision,
            "likelihoods" : model_likelihoods
        }

    
    def approx_2_int(self, num_pts, eps):

        pt_vec = 2 * (torch.rand((num_pts, 1), device=next(self.parameters()).device) - .5)

        x_pre = pt_vec - eps
        x_post = pt_vec + eps
        
        pred_pre = self(x_pre)
        pred_post = self(x_post)
        
        mgrads = (pred_post['location'] - pred_pre['location']) / (2 * eps)

        pgrads = (pred_post['scale'] - pred_pre['scale']) / (2 * eps)

        mints = mgrads.pow(2).mean(0).flatten()
        pints = pgrads.pow(2).mean(0).flatten()

      
        return({'mint': mints, 'pint': pints})
    
    def comp_grad_forward(self, x): 

        assert(x.dim() == 3)
        form = "bmi,mio->bmo"
        l_grad = None


        location, scale = x, x

        
        # apply one layer at a time and track if it is above/below threshold
        for i in range(self.num_layers):
            # don't perform activation on final layer
            if i != 0:
                location = self.activation_func(location)
                scale = self.activation_func(scale)

            # apply one layer
            location, scale = torch.einsum(form, location, self.location_weights[i]), torch.einsum(form, scale, self.scale_weights[i])

            location, scale = location + self.location_biases[i], scale + self.scale_biases[i]
            
            # track where greater than threshold
            l_flags = location > 0
            s_flags = scale > 0
            
            # don't apply activation to final layer, let all of them multiply through(?)
            if i == self.num_layers-1:
                l_flags = torch.ones_like(l_flags)
                s_flags = torch.ones_like(s_flags)
                
            if isinstance(self.activation_func, nn.LeakyReLU):

                l_weight = torch.matmul(torch.diag_embed(torch.where(l_flags, 1., 
                                                                     self.activation_func.negative_slope)),
                                        self.location_weights[i].transpose(-1, -2)).transpose(-1, -2).squeeze(-2)
                
                s_weight = torch.matmul(torch.diag_embed(torch.where(s_flags, 1., 
                                                                     self.activation_func.negative_slope)),
                                        self.scale_weights[i].transpose(-1, -2)).transpose(-1, -2).squeeze(-2)

            
            if l_grad is None:
                l_grad = l_weight
                s_grad = s_weight
            else:

                s_grad = torch.matmul(s_grad, s_weight)
                l_grad = torch.matmul(l_grad, l_weight)
            

            
        
        # chain rule for activation function on scale parameter
        if isinstance(self.scale_activation_func, nn.Softplus):
    
            s_grad = s_grad * (torch.exp(scale) * (1 + torch.exp(scale)).pow(-1)).unsqueeze(-2)

        if self.scale_activation=="exp":

            s_grad = s_grad * torch.exp(scale).unsqueeze(-2)
            
            
    
        if not self.get_inv_param():
            preact_scale = scale
            scale = (self.scale_activation_func(scale) *  (1 - 1e-8)) + 1e-8

            return {"loc_grad": l_grad, 
                    "scale_grad": s_grad, 
                    "scale": scale, 
                    "location": location}

        else:
            preact_scale = scale
            scale = (self.scale_activation_func(scale) *  (1 - 1e-8)) + 1e-8
            
            return {"loc_grad": l_grad, 
                    "scale_grad": s_grad, 
                    "precision": scale,
                    "scale": scale.pow(-.5), 
                    "preact_scale": preact_scale,
                    "location": location}
    
    # modified forward that tracks the gradient manually
    def comp_grad_fourier_forward(self, x): 

        assert(x.dim() == 3)
        form = "bmi,mio->bmo"
        l_grad = None

        transformed_x = torch.einsum(form, x, self.B_mat)
        transformed_x = torch.cat([torch.cos(transformed_x), torch.sin(transformed_x)], dim=-1)

        location, scale = transformed_x, transformed_x

        
        # apply one layer at a time and track if it is above/below threshold
        for i in range(self.num_layers):
            # don't perform activation on final layer
            if i != 0:
                location = self.activation_func(location)
                scale = self.activation_func(scale)

            # apply one layer
            location, scale = torch.einsum(form, location, self.location_weights[i]), torch.einsum(form, scale, self.scale_weights[i])

            location, scale = location + self.location_biases[i], scale + self.scale_biases[i]
            
            # track where greater than threshold
            l_flags = location > 0
            s_flags = scale > 0
            
            # don't apply activation to final layer, let all of them multiply through(?)
            if i == self.num_layers-1:
                l_flags = torch.ones_like(l_flags)
                s_flags = torch.ones_like(s_flags)
                
            if isinstance(self.activation_func, nn.LeakyReLU):

                l_weight = torch.matmul(torch.diag_embed(torch.where(l_flags, 1., 
                                                                     self.activation_func.negative_slope)),
                                        self.location_weights[i].transpose(-1, -2)).transpose(-1, -2).squeeze(-2)
                
                s_weight = torch.matmul(torch.diag_embed(torch.where(s_flags, 1., 
                                                                     self.activation_func.negative_slope)),
                                        self.scale_weights[i].transpose(-1, -2)).transpose(-1, -2).squeeze(-2)

            
            if l_grad is None:
                l_grad = l_weight
                s_grad = s_weight
            else:

                s_grad = torch.matmul(s_grad, s_weight)
                l_grad = torch.matmul(l_grad, l_weight)
            

            
        
        # chain rule for activation function on scale parameter
        if isinstance(self.scale_activation_func, nn.Softplus):
    
            s_grad = s_grad * (torch.exp(scale) * (1 + torch.exp(scale)).pow(-1)).unsqueeze(-2)
        
        if self.scale_activation == "exp":
            s_grad = s_grad * torch.exp(scale).unsqueeze(-2)
            
            
        
        top = -torch.sin(torch.einsum(form, x, self.B_mat)).unsqueeze(-2) * self.B_mat.unsqueeze(0)
        bottom = torch.cos(torch.einsum(form, x, self.B_mat)).unsqueeze(-2) * self.B_mat.unsqueeze(0)

        trans_to_x = torch.cat([top, bottom], dim=-1)


        l_grad = torch.matmul(trans_to_x, l_grad)
        s_grad = torch.matmul(trans_to_x, s_grad)
    
        if not self.get_inv_param():
            preact_scale = scale
            scale = (self.scale_activation_func(scale) *  (1 - 1e-8)) + 1e-8

            return {"loc_grad": l_grad, 
                    "scale_grad": s_grad, 
                    "scale": scale, 
                    "preact_scale": preact_scale,
                    "location": location}

        else:
            preact_scale = scale
            scale = (self.scale_activation_func(scale) *  (1 - 1e-8)) + 1e-8
            
            return {"loc_grad": l_grad, 
                    "scale_grad": s_grad, 
                    "precision": scale,
                    "preact_scale": preact_scale,
                    "scale": scale.pow(-.5), 
                    "location": location}