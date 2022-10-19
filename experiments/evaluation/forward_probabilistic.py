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
from laplace import Laplace
from tqdm import tqdm
from src.models.trainer import ProbRegressionTrainer

def add_args(parser):
    parser.add_argument('--checkpoint_metric', type=str, default="best_valid_nll", choices=["best_valid_mse", "best_valid_nll"],
                    help='Based on which metric the best model to load is chosen.')
    parser.add_argument("--ood", action="store_true", help="If true, also performs ood testing")
    return parser

if __name__ == "__main__":
    args, model_args, ds_args = parse_eval_args(add_argparser=add_args)
    print(args)

    # torch setup
    args["device"] = torch.device('cuda:' + str(args["gpu_id"]) if torch.cuda.is_available() else 'cpu')

    if args["task"] not in ["gaussian"]:
        raise ValueError("Probabilistic evaluation currently just implemented for gaussian case")

    # set training stategies to False
    args["weighted"] = False
    args["even_split"] = False
    args["splits"] = ["train"] + args["splits"] # wer need train split for Laplace estimate
    train_loader, test_loader = load_dset_electrolyte(**merge_configs(ds_args, args))

    # define the model
    N_LEADS = model_args["n_leads"]
    N_CLASSES = 1
    model_constructor = get_model(args["task"], model_args["name"], metadata=args["use_metadata"])
    model = model_constructor(input_dim=(N_LEADS, model_args["seq_length"]),
                     blocks_dim=list(zip(model_args["net_filter_size"], model_args["net_seq_lengh"])),
                     n_classes=N_CLASSES,
                     kernel_size=model_args["kernel_size"],
                     dropout_rate=model_args["dropout_rate"])
    trainer = ProbRegressionTrainer(model)
    trainer.load(args["load_path"], model_name=args["checkpoint_metric"])
    model.to(args["device"])


    # add Laplace model
    model.reformat_Laplace() # assign the forward mean to the forward method for Laplace library
    train_loader.format_Laplace() # assign dataset variable to dataset

    laplace_model = Laplace(model, "regression", subset_of_weights='last_layer', hessian_structure='full')
    laplace_model.fit(train_loader)
    log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
    hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
    for i in tqdm(range(10)):
        hyper_optimizer.zero_grad()
        neg_marglik = - laplace_model.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
        neg_marglik.backward()
        hyper_optimizer.step()

    
    # start the evaluation
    model.eval()
    def forward_dataloader(dataloader, ecg_transform=None):
        target_list = []
        prediction_list = []
        epistemic_var_list = []
        aleatoric_var_list = []
        with torch.no_grad():
            train_bar = tqdm(initial=0, leave=True, total=len(dataloader), position=0)
            for data, target in dataloader:
                target_list.append(target)
                if ecg_transform:
                    data = ecg_transform(data)
                prediction, epistemic_variance = laplace_model(data)
                aleatoric_variance = model.forward_log_var(data).exp()
                prediction_list.append(prediction)
                epistemic_var_list.append(epistemic_variance)
                aleatoric_var_list.append(aleatoric_variance)
                train_bar.update(1)
            train_bar.close()
        target_list = torch.cat(target_list, dim=0).cpu().flatten()
        prediction_list = torch.cat(prediction_list, dim=0).cpu().flatten()
        epistemic_var_list = torch.cat(epistemic_var_list, dim=0).cpu().flatten()
        aleatoric_var_list = torch.cat(aleatoric_var_list, dim=0).cpu().flatten()
        return target_list, prediction_list, epistemic_var_list, aleatoric_var_list
        

    # define the two noise transformations

    trace_len = 4096 # fixed for our experiments
    def gaussian_noise(data:torch.Tensor, size:float=1):
        return data + size * torch.randn_like(data)


    def random_mask(data:torch.Tensor, length:int):
        for single in data:
            lower = torch.randint(0, trace_len - length + 1, (1,))
            for lead in single:
                lead[lower: lower + length] = 0
        return data

    # do the actual experiments and save
    
    df_list = []
    columns = []

    df_list += list(forward_dataloader(test_loader))
    columns = ["Target", "prediction", "epistemic_variance", "aleatoric_variance"]
    model_name = os.path.normpath(args["load_path"]).split(os.sep)[-1]
    df_list = torch.stack(df_list, dim=-1)
    df = pd.DataFrame(data=df_list.tolist(), columns=columns)
    filename = os.path.join(args["load_path"], f"prediction_{ds_args['id'].lower()}_gaussian_laplace_{model_name}_split_{args['splits'][0]}.csv")
    df.to_csv(filename, index=False)

    if args["ood"]:    
        columns = ["target", "prediction", "laplace_baseline", "gaussian_baseline"]
        for noise in [0.0931, 0.2624]:
            print(f"Starting noise {noise}")
            df_list += list(forward_dataloader(test_loader, ecg_transform=lambda data: gaussian_noise(data, noise)))[1:]
            tag = f"_noise_{noise}"
            columns += ["prediction" + tag, "laplace" + tag, "gaussian" + tag]

        for mask_len in [int(percentage * trace_len) for percentage in [0.25, 0.5, 0.75]]:
            print(f"Starting mask {mask_len}")
            df_list += list(forward_dataloader(test_loader, ecg_transform=lambda data: random_mask(data, length=mask_len)))[1:]
            tag = f"_mask_{mask_len}"
            columns += ["prediction" + tag, "laplace" + tag, "gaussian" + tag]
        
        for clip_size in [12.5, 25, 37.5]:
            print(f"Starting clip {clip_size}")
            df_list += list(forward_dataloader(test_loader, ecg_transform=lambda data: torch.clamp(data, -clip_size, clip_size)))[1:]
            tag = f"_clip_{clip_size}"
            columns += ["prediction" + tag, "laplace" + tag, "gaussian" + tag]

        df_list = torch.stack(df_list, dim=-1)
        df = pd.DataFrame(data=df_list.tolist(), columns=columns)

        model_name = os.path.normpath(args["load_path"]).split(os.sep)[-1]
        filename = os.path.join(args["load_path"], f"ood_prediction_{ds_args['id']}_gaussian_laplace_{model_name}_split_{args['splits'][0]}.csv")
        df.to_csv(filename, index=False)