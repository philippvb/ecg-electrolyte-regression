import sys, os
sys.path.insert(0, os.getcwd())
from src.models import get_model, get_trainer
import torch
from src.dataset.electrolytes import load_dset_electrolyte
import torch.optim as optim
from src.utils.argparser import  parse_ecg_args, save_args
from src.utils.random import merge_configs
import numpy as np



if __name__ == "__main__":
    args, model_args, ds_args = parse_ecg_args()
    print("Args", args)

    if args["buckets_linspace"]:
        bucket_args = args["buckets_linspace"]
        args["buckets"] = np.linspace(bucket_args[0], bucket_args[1], int(bucket_args[2]) - 1).tolist()


    # Generate output folder if needed
    if not os.path.exists(args["folder"]):
        os.makedirs(args["folder"])


    save_args(args, model_args, ds_args)




    # torch setup
    torch.manual_seed(args["seed"])
    args["device"] = torch.device('cuda:' + str(args["gpu_id"]) if torch.cuda.is_available() else 'cpu')

    # get the dataset 
    if args["task"] in ["classification", "ordinal"]:
        if args["task"] == "classification":
            args["as_classification"] = True
        elif args["task"] == "ordinal":
            args["as_classification"] = True
            args["ordinal"] = True
        args["buckets"] = np.array(args["buckets"])
            
    train_loader, valid_loader = load_dset_electrolyte(**merge_configs(ds_args, args))

    # define the model
    N_LEADS = model_args["n_leads"]
    if args["task"] in ["classification", "ordinal"]:
        N_CLASSES = len(args["buckets"]) + 1 if len(args["buckets"]) > 1 else len(args["buckets"])
    else:
        N_CLASSES = 1
    model_constructor = get_model(args["task"], model_args["name"], metadata=args["use_metadata"])
    model = model_constructor(input_dim=(N_LEADS, model_args["seq_length"]),
                     blocks_dim=list(zip(model_args["net_filter_size"], model_args["net_seq_lengh"])),
                     n_classes=N_CLASSES,
                     kernel_size=model_args["kernel_size"],
                     dropout_rate=model_args["dropout_rate"])


    # define the trainer
    if (type(args["buckets"]) == np.ndarray) and (len(args["buckets"]) == 1):
        args["task"] = "binary"
    trainer = get_trainer(args["task"])(model)

    trainer.create_optimizer(optim.Adam, lr=args["lr"])

    trainer.create_scheduler(patience=args["patience"],min_lr=args["lr_factor"] * args["min_lr"],
                                                     factor=args["lr_factor"])

    if args["task"] == "gaussian":
        trainer.set_burnin(args["burnin"])

    trainer.train(args["epochs"], train_loader, valid_loader, args["min_lr"], folder=args["folder"], device=args["device"], weighted=args["weighted"])