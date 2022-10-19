import torch

def mse(pred:torch.Tensor, target:torch.Tensor, weights=None, reduction=torch.sum)->torch.Tensor:
    diff = target.flatten() - pred.flatten()
    if not reduction: # set reduction to identity function
        reduction = lambda x: x 
    if torch.is_tensor(weights):
        loss = reduction(weights.flatten() * diff * diff)
    else:
        loss = reduction(diff * diff)
    return loss


def mae(pred:torch.Tensor, target:torch.Tensor, weights=None, reduction=torch.sum)->torch.Tensor:
    diff = target.flatten() - pred.flatten()
    if not reduction: # set reduction to identity function
        reduction = lambda x: x 
    if torch.is_tensor(weights):
        wmae = reduction(weights.flatten() * torch.abs(diff))
    else:
        wmae = reduction(torch.abs(diff))
    return wmae

def gaussian_nll(pred: torch.Tensor, target: torch.Tensor, pred_log_var: torch.Tensor, weights=None, reduction=torch.mean) -> torch.Tensor:
    """Computes the sum of the batch negative log-likelihoods under a normal distribution: N(target, pred, pred_var). Scaling constants are dropped.

    Args:
        target (torch.Tensor): The target
        pred (torch.Tensor): The prediction
        pred_var (torch.Tensor): The variance of the prediction
        weights: if not None, will weight the datapoints accordingly

    Returns:
        (torch.Tensor): The sum of the negative log_likelihoods over the batch
    """
    if not reduction: # set reduction to identity function
        reduction = lambda x: x 
    mse = torch.pow(target.flatten() - pred.flatten(), 2)
    exponent = torch.exp(-pred_log_var.flatten())*mse
    # exponent = torch.clamp(exponent, max=1000)
    loss = exponent + pred_log_var.flatten()
    if torch.any(torch.isnan(loss)):
        print("NAN", exponent, pred_log_var)
        loss = torch.zeros_like(loss)
    loss = torch.clamp(loss, 10000)
    if torch.is_tensor(weights):
        loss = weights * loss
    loss = reduction(loss)
    return loss, reduction(exponent), reduction(pred_log_var)

def bce(pred: torch.Tensor, target: torch.Tensor, weights=None, reduction=torch.sum) -> torch.Tensor:
    if not reduction: # set reduction to identity function
        reduction = lambda x: x 
    target = target.reshape(pred.shape)
    pred = pred.clamp(min=-100).sigmoid()
    loss = -(target * pred.log() + (1-target) * (1 - pred).log())
    if torch.is_tensor(weights):
        loss = weights * loss
    loss = reduction(loss)
    return loss 

SIGMOID_EPS = 1e-5
def coral(pred: torch.Tensor, target: torch.Tensor, weights=None, reduction=torch.mean) -> torch.Tensor:        
    pred = pred.clamp(min=SIGMOID_EPS, max=1-SIGMOID_EPS) # clamping like in pytorch bce to avoid exploding log 
    loss = -(pred.log() * target + (1-pred).log() * (1 - target))
    if torch.is_tensor(weights):
        weights = weights.reshape(loss.shape[0], 1)
        loss *= weights
    return reduction(loss)

def accuracy(pred: torch.Tensor, target: torch.Tensor, reduction=torch.mean):
    return reduction((pred.argmax(dim=1) == target).float())