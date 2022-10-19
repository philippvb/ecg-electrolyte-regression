import sys, os
sys.path.insert(0, os.getcwd())
from src.models import get_model, get_trainer
import torch
from src.dataset.electrolytes import load_dset_electrolyte
import torch.optim as optim
from src.utils.argparser import  parse_ecg_args, parse_eval_args, save_args
from src.utils.random import merge_configs
import numpy as np
import pandas as pd



if __name__ == "__main__":
    args, model_args, ds_args = parse_eval_args()
    print(args)

    # torch setup
    args["device"] = torch.device('cuda:' + str(args["gpu_id"]) if torch.cuda.is_available() else 'cpu')


    # get the dataset 
    if args["task"] == "classification":
        args["buckets"] = np.array(args["buckets"]) # transform back to np array
    # set training stategies to False
    args["weighted"] = False
    args["even_split"] = False
    test_loader, = load_dset_electrolyte(**merge_configs(ds_args, args))


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

    trainer = get_trainer(args["task"])(model)

    trainer.load(args["load_path"])
    model.to(args["device"])

    prediction_list, target_list, weights = trainer.create_predictions(ep=-1, dataloader=test_loader, weighted=False)

    model_name = os.path.normpath(args["load_path"]).split(os.sep)[-1]
    filename = os.path.join(args["load_path"], f"prediction_{ds_args['id']}_{args['task']}_{model_name}_split_{args['splits'][0]}.csv")
    if args["normalize"]:
        df = pd.read_csv(ds_args["path_to_csv"])
        electrolyte = df[ds_args["el_col"]]
        train_size = int(args["train_split"] * (df["split"] == "train").sum())
        train_mask = np.zeros(len(df))
        train_index = np.where(df["split"] == 'train')[0][:train_size]
        train_mask[train_index] = 1
        train_mask_bool = train_mask.astype("bool")
        mean, std = np.mean(electrolyte[train_mask_bool]), np.std(electrolyte[train_mask_bool])
        with open(filename, "w") as f:
            f.write(f"Normalization:\nmean,std\n{mean},{std}\n")
    
    if args["task"] in ["classification", "ordinal"]:
        with open(filename, "w") as f:
            f.write(f"intervals,{list(args['buckets'])}\n")
    
    data = torch.cat((prediction_list, target_list), dim=-1).numpy()
    n_outputs = N_CLASSES - 1 if args["task"] == "ordinal" else N_CLASSES
    columns = [f"output_{i}" for i in range(n_outputs)] + ["target"]
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(filename, index=False, mode="a")
    