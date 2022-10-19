from typing import Tuple
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from src.dataset.batchdataloader import BatchDataloader
from torch import nn
import torch
from src.models.resnet import ProbResNet1d
from src.evaluations.utils import remove_outliers, boxplot_buckets
from src.dataset.utils import unpack_batch
import numpy as np

SCATTER_CONFIG = {"color":"black", "s":1}
AGE_LIM = (15,100)
POTASSIUM_LIM = (10, 100)
CALCIUM_LIM = (10, 40)

# higher level function for chaining
def plot_multiple(functions, dataset:BatchDataloader, model, axs, error_fun, prob=True, laplace=False, plot_parameters=SCATTER_CONFIG, x_lim=AGE_LIM, meta=False):
    if not meta:
        dataset.remove_weights()
    pred, pred_var, errors, ages = forward_summary(dataset, model, error_fun, prob=prob, meta=meta)
    data_noise = model.sigma_noise.item() if laplace else None
    for id, fun in enumerate(functions):
        fun(axs=axs[id], pred=pred, pred_var=pred_var, errors=errors, ages=ages, data_noise=data_noise, plot_parameters=plot_parameters, x_lim=x_lim)


def plot_multiple_from_stats(functions, axs, predictions, targets, errors, predictions_var=None, plot_parameters=SCATTER_CONFIG, x_lim=AGE_LIM):
    for id, fun in enumerate(functions):
        fun(axs=axs[id], pred=predictions, pred_var=predictions_var, errors=errors, ages=targets, plot_parameters=plot_parameters, x_lim=x_lim)

## -------------------------------standard-------------------------------------
# target vs error
@torch.no_grad()
def plot_age_vs_error(errors, ages, axs:Axes, plot_parameters=SCATTER_CONFIG, x_lim=AGE_LIM, **kwargs):
    # errors, ages = remove_outliers(errors, ages, n=100)
    # labels, errors = boxify_ages(ages, errors)
    # axs.boxplot(errors, labels=labels)
    # axs.set_xlabel("target Age")
    # axs.set_ylabel("Error")
    
    errors, ages = remove_outliers(errors, ages, n=10)
    axs.scatter(ages.cpu(), errors.cpu(), **plot_parameters)
    axs.set_xlabel("Target")
    axs.set_ylabel("Error")
    axs.set_xlim(x_lim[0], x_lim[1])


# predicted vs error
@torch.no_grad()
def plot_predicted_age_vs_error(pred, errors, axs:Axes, plot_parameters=SCATTER_CONFIG, x_lim=AGE_LIM, **kwargs):
    errors, pred = remove_outliers(errors, pred, n=10)
    axs.scatter(pred.cpu(), errors.cpu(), **plot_parameters)
    axs.set_xlim(25, 100)
    # axs.set_ylim(0, errors.max())
    axs.set_xlabel("Prediction")
    axs.set_ylabel("Error")
    axs.set_xlim(x_lim[0], x_lim[1])

# predicted vs target, id function
@torch.no_grad()
def plot_age_vs_predicted(pred:torch.Tensor, ages:torch.Tensor,  axs:Axes, plot_parameters=SCATTER_CONFIG, x_lim=AGE_LIM, **kwargs):
    axs.scatter(ages.cpu(), pred.cpu(), **plot_parameters)
    axs.set_xlabel("Target")
    axs.set_ylabel("Prediction")
    axs.plot(x_lim, x_lim, c="red") # identity line
    axs.set_xlim(x_lim[0], x_lim[1])
    axs.set_ylim(x_lim[0], x_lim[1])

@torch.no_grad()
def boxplot_target_vs_predicted(pred:torch.Tensor, ages:torch.Tensor,  axs:Axes, x_lim=AGE_LIM, **kwargs):
    pred, ages = remove_outliers(pred, ages, n=10)
    eps = 0.1
    buckets = torch.linspace(ages.min()-eps, ages.max(), 10)
    target_bucket_classes = torch.bucketize(ages, buckets)
    buckets[0] += eps
    predicitions_bucketized = [pred[target_bucket_classes == i] for i in range(1, len(buckets))]
    bucket_labels = [f"{round(lower.item(), 1)}/\n{round(upper.item(), 1)}" for lower, upper in zip(buckets[:-1], buckets[1:])]
    axs.boxplot(predicitions_bucketized, labels=bucket_labels, showfliers=False)
    axs.set_xlabel("Target")
    axs.set_ylabel("Prediction")
    axs.xaxis.set_tick_params(labelsize=10)
    axs.plot([1, len(buckets)-1], [ages.min(), ages.max()], c="red")



## -------------------------------calibration ---------------------------------
@torch.no_grad()
def plot_calibration(pred_var, errors, axs:Axes, plot_parameters=SCATTER_CONFIG, **kwargs):
    pred_var, errors = remove_outliers(pred_var, errors, n=10)
    axs.scatter(errors.cpu(), pred_var.exp().sqrt().cpu(), **plot_parameters)
    cat_tensor = torch.cat([errors, pred_var], dim=0)
    xy = [torch.min(cat_tensor), torch.max(cat_tensor)]
    axs.plot(xy, xy, c="red") # plot vline
    axs.set_xlabel("Error")
    axs.set_ylabel("Predicted standard deviation")

@torch.no_grad()
def plot_calibration_laplace(pred_var, errors, data_noise, axs:Axes, plot_parameters=SCATTER_CONFIG, **kwargs):
    pred_var, errors = remove_outliers(pred_var, errors, n=100)
    var = pred_var.cpu() + data_noise**2 # add the datanoise if the model
    axs.scatter(var, errors.cpu(),  **plot_parameters)
    axs.set_xlabel("Predicted variance")
    axs.set_ylabel("Error")

@torch.no_grad()
def plot_calibration_laplace_gaussian(dataset:BatchDataloader, laplace_model:nn.Module, model:ProbResNet1d, axs:Axes, error_fun=torch.nn.MSELoss, plot_parameters=SCATTER_CONFIG):
    errors_list = torch.empty(1)
    ages_list = torch.empty(1)
    var_list = torch.empty(1)
    pred_list = torch.empty(1)
    for data, target, _ in dataset:
        # forward pass
        pred = laplace_model(data)
        pred = pred[0]
        pred_var = pred[1] # the predictive variance from laplace
        data_var = model.forward_log_var(data).exp() # the estimated data variance from laplace
        print(pred_var.shape)
        print(data_var.shape)
        total_var = pred_var + data_var # add them to combine
        # append to tracking
        var_list = torch.cat((var_list, total_var.squeeze().cpu()), dim=0)
        errors = error_fun(pred, target)
        errors_list = torch.cat((errors_list, errors.squeeze().cpu()), dim=0)
        ages_list = torch.cat((ages_list, target.squeeze().cpu()), dim=0)
        pred_list = torch.cat((pred_list, pred.squeeze().cpu()), dim=0)
    # remove outliers
    # errors, pred_var = remove_outliers(errors, pred_var, n=100)
    # plot the data
    axs.scatter(var_list, errors.cpu(),  **plot_parameters)
    axs.set_xlabel("Predicted variance")
    axs.set_ylabel("Error")


# summary 

def plot_summary(axs, summary_dict:dict):
    textstr = "Summary\n"
    # extract data if tensor etc
    def extract(data):
        if torch.is_tensor(data):
            data = data.item()
        if type(data) == np.ndarray:
            data = np.around(data, 3)
        if type(data) == float:
            data = round(data, 3)
        if (type(data) == list) and (type(data[0])==float):
            data = [round(item, 2) for item in data]
        return data

    textstr += "\n".join([f"{key}: {extract(value)}" for key, value in summary_dict.items()])
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    axs.text(0, 1, textstr, transform=axs.transAxes, fontsize=14,
        verticalalignment='top', bbox=props, wrap=True)


# utils

# the forward pass required for the data
@torch.no_grad()
def forward_summary(dataset:BatchDataloader, model:torch.nn.Module, error_fun=torch.nn.MSELoss, prob=True, meta=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    errors_list = torch.empty(1)
    ages_list = torch.empty(1)
    pred_log_var_list = torch.empty(1)
    pred_list = torch.empty(1)
    for batch in dataset:
        data, target, _ = unpack_batch(batch)
       # print(data)
        pred = model(data)
        if prob:
            pred_log_var = pred[1]
            pred = pred[0] # drop var if probabilistic model
            pred_log_var_list = torch.cat((pred_log_var_list, pred_log_var.squeeze().cpu()), dim=0)
        errors = error_fun(target, pred)
        errors_list = torch.cat((errors_list, errors.squeeze().cpu()), dim=0)
        ages_list = torch.cat((ages_list, target.squeeze().cpu()), dim=0)
        pred_list = torch.cat((pred_list, pred.squeeze().cpu()), dim=0)
    return pred_list, pred_log_var_list, errors_list, ages_list



@torch.no_grad()
def plot_ages_vs_var(ages, pred_var, axs:Axes, plot_parameters=SCATTER_CONFIG, **kwargs):
    labels, pred_var = boxify_ages(ages, (0.5 *pred_var).exp())
    axs.boxplot(pred_var, labels=labels)
    # axs.set_xlim(25, 100)
    axs.set_xlabel("Target")
    axs.set_ylabel("predicted Standard Deviation")


@torch.no_grad()
def boxify_ages(ages, tensor, step_size=10):
    out_tensor = []
    out_ages = []
    for age in range(20, 100, step_size):
        mask = torch.logical_and(ages < age + step_size, ages >= age)
        out_tensor.append(torch.masked_select(tensor, mask))
        out_ages.append(f"{age}-{age+step_size-1}")
    out_ages = list(out_ages)
    return out_ages, out_tensor

def add_model_description(train_args:dict, summary:dict, keys=["normalize", "use_weights", "even_split", "use_metadata", "as_classification"]):
    for key in keys:
        if key in train_args.keys():
            summary[key] = train_args[key]
        else:
            summary[key] = False


### new functions
def plot_scatter(xvalues, yvalues, axs:Axes, xlabel, ylabel, x_lim=None, y_lim=None, plot_parameters=SCATTER_CONFIG, outliers=None):
    if outliers:
        xvalues, yvalues = remove_outliers(xvalues, yvalues, outliers=outliers)
    axs.scatter(xvalues, yvalues, **plot_parameters)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    if x_lim:
        axs.set_xlim(x_lim)
    if y_lim:
        axs.set_ylim(y_lim)

def plot_target_vs_variance(target, variance, axs:Axes, x_lim=None, y_lim=None, plot_parameters=SCATTER_CONFIG, outliers=None):
    plot_scatter(target, variance, axs, "Target", "Predicted Variance", x_lim, y_lim, plot_parameters, outliers)

def plot_target_vs_error(target, error, axs:Axes, x_lim=None, y_lim=None, plot_parameters=SCATTER_CONFIG, outliers=None):
    plot_scatter(target, error, axs, "Target", "Error", x_lim, y_lim, plot_parameters, outliers)


