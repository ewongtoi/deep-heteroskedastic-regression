import argparse
from datetime import datetime

def general_args(parser):
    group = parser.add_argument_group("General set arguments for miscelaneous utilities.")
    group.add_argument("--dont_print_args", action="store_true", help="Specify to disable printing of arguments.")


def model_args(parser):
    group = parser.add_argument_group("arguments for mean/sigma model architecures.")
    group.add_argument("--hidden_size", type=int, default=128, help="number of nodes in hidden layers")
    group.add_argument("--hidden_layers", type=int, default=2, help="number of hidden layers")
    group.add_argument("--act_func", type=str, default="sigmoid", help="activation funciton for networks")
    group.add_argument("--prec_act_func", type=str, default="softplus", help="activation funciton for final layer of precision")
    group.add_argument("--var_param", action="store_true", help="model the variance, not precision")
    group.add_argument("--diag", action="store_true", help="perform a diagonal search with given gammas")
    



def train_args(parser):
    group = parser.add_argument_group("arguments for training.")
    group.add_argument("--lr", type=float, default=0.01, help="lr for optimizer")
    group.add_argument("--lr_min", type=float, default=0.01, help="min lr for optimizer")
    group.add_argument("--lr_max", type=float, default=0.01, help="max lr for optimizer")
    group.add_argument("--epochs", type=int, default=1000, help="how long to train")
    group.add_argument("--batch_size", type=int, default=None, help="how large batch size for training")
    group.add_argument("--step_size_up", type=int, default=1000, help="how long to spend on the up half of cyclic lr")
    group.add_argument("--burnin", type=int, default=1000, help="how long warmup chain")
    group.add_argument("--clip", type=float, default=10000, help="value to clip gradients")
    group.add_argument("--base", type=float, default=8., help="base for exponential regularization")
    group.add_argument("--cycle_mode", type=str, default="triangular", help="how to cycle the optimizer step size")
    group.add_argument("--pre_trained_path", type=str, default=None, help="load in pretrained model")
    group.add_argument("--per_param_loss", action="store_true", help="loss computed on a per parameter basis")
    group.add_argument("--cont_shuff", action="store_true", help="shuffle the mean/prec sets")
    group.add_argument("--aug", action="store_true", help="augment the data for mean/variance training")
    group.add_argument("--laplace", action="store_true", help="use laplace likelihood")
    group.add_argument("--train_seed", type=int, default=1234321, help="Seed for training.")
    group.add_argument("--mean_warmup", type=int, default=1000, help="how many epochs to spend on mean fits.")
    group.add_argument("--beta_nll", action="store_true", help="use beta_nll loss function")
    group.add_argument("--mle", action="store_true", help="use mle loss function")

def dataset_args(parser):
    group = parser.add_argument_group("arguments for dataset model is fit to")
    group.add_argument("--dataset", type=str, default=None, help="which dataset to load")
    group.add_argument("--homoskedastic_noise", action="store_true", help="sets noise patetrn to homoskedastic")
    group.add_argument("--samp_size", type=int, default=252, help="sample size for sim'd")
    group.add_argument("--data_seed", type=int, default=1234321, help="Seed for datageneration.")


def logging_args(parser):
    group = parser.add_argument_group("arguments for paths to store results.")
    group.add_argument("--base_model_path", type=str, default="./", help="path to folder holding saved vals/models")

def field_theory_args(parser):
    group = parser.add_argument_group("arguments for fitting field theory")
    group.add_argument("--lr", type=float, default=0.01, help="lr for optimizer")
    group.add_argument("--lr_min", type=float, default=0.01, help="min lr for optimizer")
    group.add_argument("--lr_max", type=float, default=0.01, help="max lr for optimizer")
    group.add_argument("--epochs", type=int, default=1000, help="how long to train")
    group.add_argument("--step_size_up", type=int, default=1000, help="how long to spend on the up half of cyclic lr")
    group.add_argument("--start_factor", type=float, default=0.05, help="linear increase start factor")
    group.add_argument("--total_iters", type=int, default=1000, help="how long warmup")
    group.add_argument("--clip", type=float, default=10000, help="value to clip gradients")
    group.add_argument("--base", type=float, default=8., help="base for exponential regularization")
    group.add_argument("--cycle_mode", type=str, default="triangular", help="how to cycle the optimizer step size")
    group.add_argument("--opt_scheme", type=str, default=None, help="optimization scheme")
    group.add_argument("--train_seed", type=int, default=1234321, help="Seed for training.")
    group.add_argument("--noisy_y", action="store_true", help="add in noise on the observed y's")

def int_args(parser):
    group = parser.add_argument_group("arguments for mc ints")
    group.add_argument("--base_model_path", type=str, default="./", help="path to folder holding saved vals/models")
    group.add_argument("--model_ext", type=str, default="./", help="per folder path to model")
    group.add_argument("--stats_ext", type=str, default="./", help="per folder path to train stats")
    group.add_argument("--wts_ext", type=str, default="./", help="per folder path to trained model weights")
    group.add_argument("--model_count", type=int, default=0, help="how many models to load")
    group.add_argument("--plot_path", type=str, default="./", help="path to save plots")
    group.add_argument("--data_path", type=str, default=None, help="path to data")



def print_log(*args):
    print("[{}]".format(datetime.now()), *args)

def print_args(args):
    max_arg_len = max(len(k) for k, v in args.items())

    key_set = sorted([k for k in args.keys()])
    for k in key_set:
        v = args[k]
        print_log("{} {} {}".format(
            k,
            "." * (max_arg_len + 3 - len(k)),
            v,
        ))

def get_args():

    parser = argparse.ArgumentParser()

    general_args(parser)
    model_args(parser)
    train_args(parser)
    logging_args(parser)
    dataset_args(parser)

    args = parser.parse_args()


    if not args.dont_print_args:
        print_args(vars(args))


    return args

def get_ft_args():

    parser = argparse.ArgumentParser()

    general_args(parser)
    field_theory_args(parser)
    logging_args(parser)
    dataset_args(parser)


    args = parser.parse_args()


    if not args.dont_print_args:
        print_args(vars(args))


    return args

def get_pp_args():

    parser = argparse.ArgumentParser()

    general_args(parser)
    field_theory_args(parser)
    logging_args(parser)
    dataset_args(parser)


    args = parser.parse_args()


    if not args.dont_print_args:
        print_args(vars(args))


    return args

def get_int_args():

    parser = argparse.ArgumentParser()

    general_args(parser)
    int_args(parser)
    dataset_args(parser)


    args = parser.parse_args()


    if not args.dont_print_args:
        print_args(vars(args))


    return args