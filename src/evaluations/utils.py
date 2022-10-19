import torch

@torch.no_grad()
def reg_as_class_metrics(prediction:torch.Tensor, target:torch.Tensor, thresholds:torch.Tensor=None):
    """Computes the true positive rate and false positive rate for a regression output.

    Args:
        prediction (torch.Tensor): The regression predictions of the model
        target (torch.Tensor): The classification targets
        thresholds (torch.Tensor, optional): The thresholds to use to classifiy the outputs. If None, all predictions are used as thresholds. Defaults to None.

    Returns:
        tuple(List[float], List[float]): (True positive rate, False positive rate)
    """
    if not torch.all(target.unique() == torch.tensor([0, 1]).to(target.device)):
        raise ValueError(f"The targets do not seem to contain the labels 0,1 but instead contain {target.unique()}.")

    if torch.is_tensor(thresholds):
        thresholds, _ = thresholds.sort() # sort in ascending order
    else:
        thresholds = prediction.unique() # take all values as boundary
    thresholds = thresholds.flip(0) # flip to descending order for right order of tp and fp
    tpr = []
    fpr = []
    precision = []
    recall = []
    pos_index = target.bool()
    neg_index = pos_index.logical_not()
    for t in thresholds:
        pos_prediction = (prediction >= t)
        neg_prediction = pos_prediction.logical_not()
        tp = (pos_prediction[pos_index]).sum()
        fp = (pos_prediction[neg_index]).sum()
        tn = (neg_prediction[neg_index]).sum()
        fn = (neg_prediction[pos_index]).sum()
        tpr.append((tp/(tp+fn)).item())
        fpr.append((fp/(fp+tn)).item())
        precision.append((tp/(tp+fp)).item())
        recall.append((tp/(tp+fn)).item())
    tpr = torch.tensor(tpr)
    fpr = torch.tensor(fpr)
    precision = torch.tensor(precision)
    recall = torch.tensor(recall)
    return tpr, fpr, precision, recall


def boxplot_buckets(diff, elec, buckets):
    indices = torch.bucketize(diff, buckets)
    return [elec[indices==i] for i in range(len(buckets))]

def remove_outliers(x:torch.Tensor, *tensors, n:int) -> torch.Tensor:
    kth_smallest, _ = torch.kthvalue(x, n)
    kth_largest, _ = torch.kthvalue(x, len(x)-n)
    mask =  torch.logical_and(x < kth_largest, x > kth_smallest)
    return [torch.masked_select(x, mask)] + [torch.masked_select(t, mask) for t in tensors] # apply mask to all

def average_precision(precision:torch.Tensor, recall:torch.Tensor):
    weight = recall[1:] - recall[:-1]
    weight = torch.cat([recall[0].unsqueeze(dim=0), weight]) # the first value is simply the first precision
    return (weight * precision).sum()