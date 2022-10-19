import torch
import numpy as np
import h5py
import pandas as pd
from scipy import signal as sgn
from src.dataset.batchdataloader import BatchDataloader
from src.dataset.utils import compute_weights, remove_baseline_filter


MIN_TRAIN_SET_SIZE = 1000 # the minimum size of the dataset before weight computation could become instable
BASELINE_FILTER_400 = remove_baseline_filter(400)

def get_single_trace(args, id):
    # get traces
    f = h5py.File(args["path_to_traces"], 'r')
    trace = f[args["traces_dset"]][id]
    if trace.shape[1] < trace.shape[0]:
        trace = trace.T # transpose if wrong order
    return trace

def load_dset_brazilian(args, use_weights=True, map_to_swedish=False, device="cpu", return_ids=False):
    """Loads the dataset given in brazilian format, that is hdf5 for traces and csv for ages. Splits into train and valid

    Args:
        args (dict): The arguments for loading, normally coming from the config.json. Following parameters are needed:
            - path_to_csv: Path to the age csv file
            - ids_col: Name for the (patient) ids column in the csv
            - age_col: Name for the age column in the csv
            - path_to_traces: Path to the traces hdf5 file
            - traces_dset: Name for the traces column 
            - ids_dset: Name for the (patient) ids column in the traces file
            - train_split: How much of the dataset to use for training, on a scale from 0 to 1 (dataset_subset)
            - valid_split: How much of the dataset to use for validation, on a scale from 0 to 1 (n_valid)
        use_weights (bool, optional): Wether to add the weights to the train_loader (always in valid_loader). Defaults to True.
        device (str): The device to push the data to

    Returns:
        tuple(BatchDataloader, BatchDataloader): The train and validation set
    """
    # Get age data in csv
    df = pd.read_csv(args["path_to_csv"], index_col=args["ids_col"])

    # get traces
    f = h5py.File(args["path_to_traces"], 'r')
    traces = f[args["traces_dset"]]


    # if we have ids in dset, check that indexes match
    if args["ids_dset"]:
        h5ids = f[args["ids_dset"]]
        df = df.reindex(h5ids, fill_value=False, copy=True)
    else:
        print("ids_dset col not given, thus assuming we dont need to match ages to traces")

    ages = np.array(df[args["age_col"]], dtype=np.float32) # somehow we need change the type of ages to get a proper matching

    # check dimensions
    if len(ages) != len(traces):
        print("Warning: Length between ages and traces doesn't seem to match")
        if len(ages) < len(traces):
            raise ValueError("Ages csv contains less datapoints than traces")
    
    # Train/ val split
    total_length = len(traces)
    valid_mask = np.arange(len(df)) > (total_length * (1 - args["valid_split"]))
    # check total split:
    if args["valid_split"] + args["train_split"] > 1:
        raise ValueError("Sum of train and valid split is larger than 1")
    # take subset if needed, else just take whole remaining
    if args["train_split"] !=1:
        train_mask = np.arange(len(df)) <= args["train_split"] * total_length
    else:
        train_mask = ~valid_mask

    def map_swedish_and_filter(traces):
        # remove leads
        traces = map_brazilian_to_swedish(traces)
        # filter frequencies
        traces = sgn.sosfiltfilt(BASELINE_FILTER_400, traces, padtype='constant', axis=-1)
        # scale to swedish data
        traces /= 2
        return traces

    # define the mapping
    mapping = map_swedish_and_filter if map_to_swedish else None

    # define dataloader
    weights = compute_weights(ages)
    # for validation we always want weights
    if args["train_split"] * total_length < MIN_TRAIN_SET_SIZE:
        print("Warning: Train dataset size seems very small, weights could be inaccurate")
    valid_loader = BatchDataloader(traces, ages, weights, bs=args["batch_size"], mask=valid_mask, transpose=True, traces_mapping=mapping, device=device)
    # train set
    if use_weights:
        train_loader = BatchDataloader(traces, ages, weights, bs=args["batch_size"], mask=train_mask, transpose=True, traces_mapping=mapping, device=device)
    else:
        train_loader = BatchDataloader(traces, ages, bs=args["batch_size"], mask=train_mask, transpose=True, traces_mapping=mapping, device=device)
    if return_ids:
        return train_loader, valid_loader, h5ids 
    else:
        return train_loader, valid_loader

def load_dset_swedish(args, use_weights=True, device="cpu"):
    """Loads the dataset given in swedish format, that is hdf5 for both traces and ages. Splits into train and valid

    Args:
        args (dict): The arguments for loading, normally coming from the config.json. Following parameters are needed:
            - path_to_traces: Path to the traces hdf5 file
            - age_col: Name for the age column 
            - traces_dset: Name for the traces column
            - train_split: How much of the dataset to use for training, on a scale from 0 to 1 (dataset_subset)
            - valid_split: How much of the dataset to use for validation, on a scale from 0 to 1 (n_valid)
        use_weights (bool, optional): Wether to add the weights to the dataset. Defaults to True.
        device (str): The device to push the data to

    Returns:
        tuple(BatchDataloader, BatchDataloader): The train and validation set
    """
    f = h5py.File(args["path_to_traces"], 'r')
    traces = f[args["traces_dset"]]
    ages = f[args["age_col"]]
    n_datapoints = len(traces)
    
    # Train/ val split
    if args["valid_split"] + args["train_split"] > 1:
        raise ValueError("Sum of train and valid split is larger than 1")

    valid_mask = np.arange(n_datapoints) > (n_datapoints * (1 - args["valid_split"]))
    # take subset if needed, else just take whole remaining
    if args["train_split"] !=1:
        train_mask = np.arange(n_datapoints) <= args["train_split"] * n_datapoints
    else:
        train_mask = ~valid_mask

    # for valid we always want weights
    if args["train_split"] * n_datapoints < MIN_TRAIN_SET_SIZE:
        print("Warning: Train dataset size seems very small, weights could be inaccurate")
    weights = compute_weights(ages)
    valid_loader = BatchDataloader(traces, ages, weights, bs=args["batch_size"], mask=valid_mask, device=device)
    # train loader
    if use_weights:
        train_loader = BatchDataloader(traces, ages, weights, bs=args["batch_size"], mask=train_mask, device=device)
    else:
        train_loader = BatchDataloader(traces, ages, bs=args["batch_size"], mask=train_mask, device=device)

    return train_loader, valid_loader
        


#-------------- Mappings ----------------------------

def map_brazilian_to_swedish(traces:torch.Tensor) -> torch.Tensor:
    """Maps the ecg traces from the brazilian to swedish format

    Args:
        traces (torch.Tensor): Traces in shape n_datapoints x timesteps x 12 (leads)

    Returns:
        torch.Tensor: reformatted Traces in shape n_datapoints x timesteps x 8 (leads)
    """
    # we have the leads in the following order:
    # DI, DII, DIII, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
    # and we want to drop III, aVR, aVL, aVF, so drop cols 3-5
    indices = np.array([0, 1, 6, 7, 8, 9, 10, 11])
    # maybe we need to reorder columns, but up to now doesnt seem like it, this is from antonios file
    # ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    return np.take(traces, indices, axis=1)