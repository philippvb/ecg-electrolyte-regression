import numpy as np
import h5py
import pandas as pd
from src.dataset.batchdataloader import BatchDataloader, EvenSamplingBatchDataloader
from src.dataset.utils import compute_weights_bucketized, check_mutual_exclusive, classes_to_levels


def load_dset_electrolyte(
                path_to_csv:str,
                path_to_traces:str,
                traces_dset:str,
                el_col:str,
                train_split:float=0.9,
                valid_split:float=0.1,
                device:str="cpu",
                weighted:bool=False,
                buckets:np.ndarray=None,
                even_split:bool=False,
                as_classification:bool=False,
                splits=["train", "val"],
                target_scaling:float=1,
                batch_size:int=32,
                max_weight=np.inf,
                use_metadata:bool=False,
                normalize:bool=False,
                ordinal:bool=False,
                log_transform:bool=False,
                additional_masks = None,
                *args,
                **kwargs
                ):
    """Load the electrolyte dataset

    Args:
        path_to_csv (str): Path to the csv containing the target values and metadata
        path_to_traces (str): Path to the h5 file containing the traces.
        traces_dset (str): Identifier for traces data in h5 file.
        el_col (str): Identifier for target electrolytes in csv file.
        train_split (float, optional): Size of training split. Defaults to 0.9.
        valid_split (float, optional): Size of validation split. Defaults to 0.1.
        device (str, optional): Device to push the dataset to. Defaults to "cpu".
        use_weights (bool, optional): If true, adds weights to the training set. Defaults to False.
        buckets (np.ndarray, optional): If given, computes which of the buckets/intervals each of the target electrolytes belongs to. Defaults to None.
        even_split (bool, optional): If set to true, returns even number of samples from each class as defined by buckets in the batches. Defaults to False.
        as_classification (bool, optional): If set to true, uses the classes defined by the buckets as targets. Defaults to False.
        splits (list, optional): List of strings defining which dataloaders containing corresponding splits to return, options are ["train", "val", "random_test", "temporal_test"]. Defaults to ["train", "val"].
        target_scaling (float, optional): Wether the targets should get scaled, results in target = targets_old * target_scaling. Defaults to 1.
        batch_size (int, optional): Defaults to 32.
        max_weight (_type_, optional): Clips the weights to the specified value, only has effect when use_weights=True. Defaults to np.inf.
        use_metadata (bool, optional): Wether to include the metadata ["sex", "age"]. Defaults to False.
        normalize (bool, optional): If true, normalizes the the targets by target = (value - mean)/sd. Defaults to False.
        ordinal (bool, optional): If true, reformulates the problem as ordinal regression by transforming the targets to the needes format using classes according to "buckets". Defaults to False.
        log_transform (bool, optional): If true, log transforms the targets. Defaults to False.
        additional_masks (List[callable], optional): If passed, restricts the training data (not test) to only examples for which every callable returns True given the target data. Defaults to None.

    Returns:
        list of BatchDataloader: Dataloaders, one for each split defined by the parameter "splits".
    """

    # Get age data in csv
    df = pd.read_csv(path_to_csv)
    electrolyte = np.array(df[el_col], dtype=np.float32)

    # get traces
    f = h5py.File(path_to_traces, 'r')
    traces = f[traces_dset]

    # train/val split

    # check total split:
    if train_split + valid_split > 1:
        raise ValueError("Sum of train and valid split is larger than 1")

    total_length = (df["split"] == "train").sum()
    val_size = int((total_length * (1 - valid_split)))
    train_size = int(train_split * total_length)

    masks = {}

    # create val set
    valid_mask = np.zeros(len(df))
    valid_index = np.where(df["split"] == 'train')[0][val_size:] # take end of dataset as val split (IMPORTANT: Assumes data is unordered, otherwise might produce weird results)
    valid_mask[valid_index] = 1
    masks["val"] = valid_mask

    # create train set
    train_mask = np.zeros(len(df))
    train_index = np.where(df["split"] == 'train')[0][:train_size] # take start of dataset as train split (IMPORTANT: Assumes data is unordered, otherwise might produce weird results)
    train_mask[train_index] = 1
    masks["train"] = train_mask

    # test splits, even though maybe not using create to ensure that we dont use test data for training

    masks["random_test"] = np.array(df["split"] == 'random test')
    masks["temporal_test"] = np.array(df["split"] == 'temporal test')

    if additional_masks:
        mask_array = [m(electrolyte=electrolyte, mask=train_mask) for m in additional_masks]
        masks["train"] = np.logical_and.reduce([masks["train"]] + mask_array)
        mask_array = [m(electrolyte=electrolyte, mask=valid_mask) for m in additional_masks]
        masks["val"] = np.logical_and.reduce([masks["val"]] + mask_array)

    # chech that the masks dont overlap
    check_mutual_exclusive(masks)

    # from now on masks are fixed
    train_mask_bool = masks["train"].astype("bool")

    if use_metadata:
        ages = np.expand_dims(np.array(df["ecg_age"], dtype=np.float32), axis=-1)
        ages = (ages - np.mean(ages[train_mask_bool]))/np.std(ages[train_mask_bool]) # normalize ages to similar range as sex
        sex =np.expand_dims(np.array(df["is_male"], dtype=np.float32), axis=-1)
    else:
        ages, sex = None, None

    if log_transform:
        print("Log transforming the electrolyte")
        electrolyte = np.log(electrolyte)

    if normalize:
        normalize = np.mean(electrolyte[train_mask_bool]), np.std(electrolyte[train_mask_bool])
    else:
        normalize = None

    loaders = []
    for name in splits:
        loaders.append(generate_dataloader(
                                traces=traces,
                                electrolyte=electrolyte,
                                mask_name=name,
                                mask=masks[name],
                                device=device,
                                use_weights=weighted,
                                buckets=buckets,
                                even_split=even_split,
                                as_classification=as_classification,
                                batch_size=batch_size,
                                target_scaling=target_scaling,
                                max_weight=max_weight,
                                normalize=normalize,
                                ages=ages,
                                sex=sex,
                                ordinal=ordinal
                                ))
        print("")
    
    return loaders


def generate_dataloader(
                traces,
                electrolyte:np.ndarray,
                mask_name:str,
                mask:np.ndarray,
                device:str="cpu",
                use_weights:bool=False,
                buckets:np.ndarray=None,
                even_split:bool=False,
                as_classification:bool=False,
                target_scaling:float=1,
                batch_size:int=32,
                max_weight:float=np.inf,
                normalize:tuple=None,
                ages:np.ndarray=None,
                sex:np.ndarray=None,
                ordinal:bool=False
                ):
    """Generates the dataloaders.

    For all args, see "load_dset_electrolyte" (above).
    Args:
        mask_name (str): The name of the mask which is used as an identifier for the dataloader.
        mask (np.ndarray): The mask to apply to the input data
        max_weight (float, optional): _description_. Defaults to np.inf.
        normalize (tuple, optional): If given as tuple of (mean, sd), appplies normalization to the targets. Defaults to None.
        ages (np.ndarray, optional): If given, adds ages metadata to each batch. Defaults to None.
        sex (np.ndarray, optional): If given, adds sex metadata to each batch. Defaults to None.

    Returns:
        BatchDataloader
    """

    print("Creating split for mask", mask_name)

    # discretize the targets
    if type(buckets) is np.ndarray:
        print("Bucketizing the electrolyte to", buckets)
        electrolyte_buckets = np.digitize(electrolyte, buckets) # discretize
        if len(np.unique(electrolyte_buckets)) == 1:
            print("WARNING: Seems like bucketizing lead to only one class")


    # add the weights
    mask_bool = mask.astype("bool")
    class_weights = None
    if use_weights:
        if even_split:
            raise ValueError("Can't use weights and use even split.")
        mask_weights, class_weights = compute_weights_bucketized(electrolyte[mask_bool], buckets=buckets, max_weight=max_weight, return_class_weights=True)
        weights = np.zeros_like(electrolyte)
        weights[mask_bool] = mask_weights
        print(f"Use weighted training, min weight: {mask_weights.min()}, max weight: {mask_weights.max()}")

    # reformulate as classification, use buckets as classes
    if as_classification:
        print("Reformulating as classification")
        electrolyte = electrolyte_buckets
        if target_scaling != 1:
            print("WARNING: Target scaling is not 1, however using classification thus setting scaling to 1")
            target_scaling = 1
        if ordinal:
            electrolyte = classes_to_levels(electrolyte, len(buckets))

    if normalize:
        if target_scaling != 1 and not (type(buckets) is np.ndarray):
            print(f"WARNING: Using normalization, however set target scaling to {target_scaling}. Therefore ignoring target scaling.")
            target_scaling = 1
        if type(buckets) is np.ndarray:
            print(f"WARNING: Using buckets, however also tried to normalize, therefore ignoring normalization")
        else:
            mean, std = normalize
            print("Data mean and std before normalization is", np.mean(electrolyte[mask_bool]), np.std(electrolyte[mask_bool]))
            electrolyte = (electrolyte - mean)/std
            print("Data mean and std after normalization is", np.mean(electrolyte[mask_bool]), np.std(electrolyte[mask_bool]), "(can be not equal to 0,1 for val and test splits)")

    if (len(electrolyte.shape) < 2) and not as_classification:
        # for regression, expand dims
        electrolyte = np.expand_dims(electrolyte, axis=-1)

    if even_split:
        print("Using even split")
        electrolyte = electrolyte_buckets
        if not (type(buckets) is np.ndarray):
            raise ValueError("WARNING: Seems like you want to train with an even split, but didn't specify buckets.")

    # build the data, order is data (including meta), target, (weights)
    data = [traces]
    if type(ages) is np.ndarray:
        data.append(ages)
    if type(sex) is np.ndarray:
        data.append(sex)
    data.append(electrolyte)
    if use_weights:
        data.append(weights)
 
    # if even split, choose dataloader accordingly
    if even_split:
        dataloader_class = EvenSamplingBatchDataloader
    else:
        dataloader_class = BatchDataloader

    return dataloader_class(*data,
                    bs=batch_size,
                    mask=mask,
                    transpose=True,
                    traces_mapping=None,
                    device=device,
                    target_scaling=target_scaling,
                    name=mask_name,
                    weighted=use_weights,
                    class_weights=class_weights)