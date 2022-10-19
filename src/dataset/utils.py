from typing import List, Tuple
import numpy as np
from scipy import signal as sgn
import torch

def check_mutual_exclusive(masks:dict):
    """Checks that the masks do not overlap (are mutually exclusive).

    Args:
        masks (dict): Dictionary containing all the masks

    Raises:
        ValueError: If the masks are overlapping
    """
    for name1, m1 in masks.items():
        for name2, m2 in masks.items():
            if name1 == name2:
                continue
            else:
                if np.any((m1 + m2) > 1):
                    raise ValueError("Seems like the masks are overlapping!")


def unpack_batch(batch:List[torch.Tensor], weighted=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpacks a batch.

    Args:
        batch (List[torch.Tensor]):
        weighted (bool, optional): If False, just adds None for weights. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: input, target, weights
    """
    if weighted:
        data, target, weights = batch[:-2], batch[-2], batch[-1]
    else:
        data, target, weights = batch[:-1], batch[-1], None
    if len(data) == 1:
        data = data[0]
    return data, target, weights



### Target transformation
def reformulate_as_classification(target:np.ndarray, buckets:np.ndarray) -> np.ndarray:
    """Reformulates the continuous targets as discrete targets for regression problem
    """
    classes = np.digitize(target, buckets, right=True)
    return classes

def classes_to_levels(classes, n_classes=None):
    n_classes = n_classes if n_classes else np.max(classes)
    levels = np.arange(n_classes)
    levels = np.vstack([levels]*len(classes))
    levels = (levels < np.array([classes]*n_classes).T).astype(int)
    return levels


### ECG transformation
    
def remove_baseline_filter(sample_rate:int):
    """Create a baselinefilter which can be used to remove baseline noise from ecgs.
    """
    fc = 0.8  # [Hz], cutoff frequency
    fst = 0.2  # [Hz], rejection band
    rp = 0.5  # [dB], ripple in passband
    rs = 40  # [dB], attenuation in rejection band
    wn = fc / (sample_rate / 2)
    wst = fst / (sample_rate / 2)
    filterorder, aux = sgn.ellipord(wn, wst, rp, rs)
    sos = sgn.iirfilter(filterorder, wn, rp, rs, btype='high', ftype='ellip', output='sos')
    return sos


### Weights computation

def compute_weights(target:np.ndarray, max_weight:float=np.inf) -> np.ndarray:
    """Computes the weights for discrete targets. Taken from https://github.com/antonior92/ecg-age-prediction.

    Args:
        target (np.ndarray): The target to compute weights of.
        max_weight (float, optional): The maximum value that the weights that gets clipped to. Defaults to np.inf.

    Returns:
        np.ndarray: _description_
    """
    _, inverse, counts = np.unique(target, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    normalized_weights = weights / sum(weights)
    w = len(target) * normalized_weights
    # Truncate weights to a maximum
    if max_weight < np.inf:
        w = np.minimum(w, max_weight)
        w = len(target) * w / sum(w) # IMPORTANT: to keep the dataset size constant (from a loss perspective, we normalize again, however this changes the max weight again)
    return w

def compute_weights_bucketized(target:np.array, buckets:np.array, max_weight:float=np.inf, return_class_weights:bool=False) -> np.ndarray:
    """Computes the weights for continuous targets by first bucketizing the target, then analogous to "compute_weights" above.

    Args:
        target (np.array): The target to compute weights of.
        buckets (np.array): Buckets/Intervals used to discretize the targets with.
        max_weight (float, optional): The maximum value that the weights that gets clipped to. Defaults to np.inf.
        return_class_weights (bool, optional): If true, additionally returns the weights by class. Defaults to False.

    Returns:
        np.ndarray: Array of corresponding weight for each target.
    """
    target_buckets = np.digitize(target, buckets, right=True)
    _, inverse, count = np.unique(target_buckets, return_inverse=True, return_counts=True)
    weights = 1/count[inverse]
    if return_class_weights:
        class_weights = 1/count
        class_weights /= sum(weights)
        class_weights *= len(weights)
    # normalize to 1
    weights /= sum(weights)
    # scale to dataset size
    weights *= len(weights)
    # cut inf off and recompute
    if max_weight < np.max(weights):
        weights = np.minimum(weights, max_weight)
        if return_class_weights:
            class_weights = np.minimum(class_weights, max_weight)
            class_weights = len(target) * class_weights / sum(weights)
        weights = len(target) * weights / sum(weights)
    if return_class_weights:
        return weights, class_weights
    else:
        return weights


### Even split creation

def create_even_split_with_discard(targets:np.array, boundaries:np.array) -> np.array:
    """First puts the targets into buckets and then discards random elements from all buckets so that all buckets have the same number of elements

    Args:
        targets (np.array): The data to split
        boundaries (np.array): The boundaries according to which the dataset should be bucketized

    Raises:
        ValueError: If there are less datapoints in a class than n_datapoints/classes or more classes than datapoints

    Returns:
        np.array: The mask
    """
    # first put targets into buckets and count
    targets = targets.flatten()
    target_bucketized = np.digitize(targets, boundaries)
    classes, counts = np.unique(target_bucketized, return_counts=True)
    per_class_examples = min(counts)
    # now take random samples from the class and put in batch mask
    mask = np.zeros_like(targets)
    for c in classes:
        mask[np.random.choice(np.argwhere(target_bucketized==c).flatten(), per_class_examples, replace=False)] = 1
        # mask[np.argwhere(target_bucketized==c).flatten()[:per_class_examples]] = 1# not random but from start
    return mask

def create_even_split(targets:np.array, boundaries:np.array, n_datapoints:int) -> np.array:
    """First puts the targets into buckets and then selects a given number of elements randomly per split, return as mask.

    Args:
        targets (np.array): The data to split
        boundaries (np.array): The boundaries according to which the dataset should be bucketized
        n_datapoints (int): Total number of datapoints

    Raises:
        ValueError: If there are less datapoints in a class than n_datapoints/classes or more classes than datapoints

    Returns:
        np.array: The mask
    """
    targets = targets.flatten()
    # first put targets into buckets and count
    target_bucketized = np.digitize(targets, boundaries)
    classes, counts = np.unique(target_bucketized, return_counts=True)
    # convert to per class examples and check for validity
    per_class_examples = int(n_datapoints/len(classes))
    if per_class_examples < 1:
        raise ValueError("The dataset contains more classes than the desired dataset size")
    if np.any(counts < per_class_examples):
        raise ValueError("Some classes contain to few examples to get to the desired class count")
    # now take random samples from the class and put in batch mask
    mask = np.zeros_like(targets)
    for c in classes:
        mask[np.random.choice(np.argwhere(target_bucketized==c).flatten(), per_class_examples, replace=False)] = 1
        # mask[np.argwhere(target_bucketized==c).flatten()[:per_class_examples]] = 1# not random but from start
    return mask

def create_split_by_condition(data:np.array, cond_list, per_cond_counts:int) -> np.array:
    """Creates a dataset mask which contains an equal number of elements for each condition provided

    Args:
        data (np.array): The data
        cond_list (List): A list of functions which evaluate each array element to true or false
        per_cond_counts (int): number of examples to pick per condition

    Returns:
        np.array: The mask for the dataset
    """
    data = data.flatten()
    mask = np.zeros_like(data)
    for cond in cond_list:
        mask[np.random.choice(np.argwhere(cond(data)).flatten(), per_cond_counts, replace=False)] = 1
    return mask