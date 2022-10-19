import argparse
from warnings import warn    
import json
import os
import numpy as np

from src.utils.random import merge_configs


def positive_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("%s is an invalid, requires positive int value" % value)
    return value

def parse_eval_args(add_argparser = None):
    parser = argparse.ArgumentParser(add_help=True,
        description='Evaluate model to predict electrolyte concentrations from ecg tracing.')
    parser.add_argument('--model', help="Path to model to evaluate")
    parser.add_argument('--splits', type=str, default="random_test", choices=["random_test", "temporal_test", "train", "val"],
                        help='The split to use for validation')

    if add_argparser:
        parser = add_argparser(parser)
    args, unk = parser.parse_known_args()
    args = vars(args)

    with open(os.path.join(args["model"], "model.json"), "r") as f:
        model_args = json.load(f)
    
    with open(os.path.join(args["model"], "dataset.json"), "r") as f:
        ds_args = json.load(f)

    with open(os.path.join(args["model"], "train_config.json"), "r") as f:
        train_args = json.load(f)

    train_args = merge_configs(train_args, args)
    train_args["splits"] = [args["splits"]]
    train_args["load_path"] = args["model"]
    
    return train_args, model_args, ds_args



def parse_ecg_args(m=None, d=None, t=None, o=None, add_argparser = None) -> dict:
    parser = argparse.ArgumentParser(add_help=True,
        description='Train model to predict electrolyte concentrations from ecg tracing.')

    general_parser = parser.add_argument_group("General")
    general_parser.add_argument('--folder', type=str, default="./model/", help="Path to store the output at (eg. model, evaluations)")
    general_parser.add_argument('--gpu_id', type=positive_int, default=0, help="Id of the gpu to use.")
    general_parser.add_argument('--m', help="Path to the json config file of the model")
    general_parser.add_argument('--d', help="Path to the json config file of the dataset")
    general_parser.add_argument("--seed", type=int, default=np.random.randint(0, 10000), help="Seed to use for pytorch.")

    optimizer_parser = parser.add_argument_group("Optimizer")
    optimizer_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate to use for the optimizer")
    optimizer_parser.add_argument("--patience", type=positive_int, default=7, help="Patience for lr scheduler")
    optimizer_parser.add_argument("--lr_factor", type=float, default=0.1, help="Factor to decrease lr by.")
    optimizer_parser.add_argument("--min_lr", type=float, default=1e-7, help="Minimum lr to stop.")

    dset_parser = parser.add_argument_group("Dataset")
    dset_parser.add_argument("--train_split", type=float, default=0.9, help="Amount of data to use for train split.")
    dset_parser.add_argument("--valid_split", type=float, default=0.1, help="Amount of data to use for validation split.")
    dset_parser.add_argument("--normalize", action="store_true", help="Normalizes the targets to mean 0 and variance 1")
    dset_parser.add_argument("--log_transform", action="store_true", help="Log transforms the target.")
    dset_parser.add_argument("--batch_size", type=positive_int, default=32, help="Batch size for training.")

    train_parser = parser.add_argument_group("Training")
    train_parser.add_argument("--task", default="regression", choices=["regression", "classification", "ordinal", "gaussian"], help="Method to use for training")
    train_parser.add_argument("--epochs", type=positive_int, default=30, help="Number of epochs to run training for.")
    train_parser.add_argument("--buckets", type=float, nargs="+", default=None, help="If given, uses these buckets/intervals to discretize targets for example for classification or weighting")
    train_parser.add_argument("--buckets_linspace", type=float, nargs=3, default=None, help="Constructs the buckets in linspace(lower, upper, steps) by passing the arguments in that order")
    train_parser.add_argument("--weighted", action="store_true", help="Use weighted training, requires specification of buckets")
    train_parser.add_argument("--use_metadata", action="store_true", help="Additionally uses age and sex metadata as input to the model.")
    train_parser.add_argument("--even_split", action="store_true", help="Use even split of training data, requires specification of buckets")
    train_parser.add_argument("--burnin", type=int, default=0, help="Set the burnin for Gaussian models.")

    if add_argparser:
        parser = add_argparser(parser)
    args, unk = parser.parse_known_args()
    args = vars(args)
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    # load the model dict
    model_json = m if m else args["m"]
    with open(model_json, "r") as f:
        model_args = json.load(f)
    
    # load the dataset dict
    dset_json = d if d else args["d"]
    with open(dset_json, "r") as f:
        ds_args = json.load(f)

    return args, model_args, ds_args


def save_args(args, model_args, ds_args):
    folder = args["folder"]
    with open(os.path.join(folder, 'model.json'), 'w') as f:
        json.dump(model_args, f, indent='\t')
    with open(os.path.join(folder, 'train_config.json'), 'w') as f:
        json.dump(args, f, indent='\t')
    with open(os.path.join(folder, 'dataset.json'), 'w') as f:
        json.dump(ds_args, f, indent='\t')
