import numpy as np
import torch
import math


class BatchDataloader:
    def __init__(self, *tensors, mask:np.ndarray, bs=32, transpose:bool=False, traces_mapping=None, device:str="cpu", target_scaling:float=1, name:str="Undefined", weighted:bool=False, class_weights=None):
        """Initializes the batchdataloader.

        Args:
            * tensors: The dataset given as a set of tensors (just need to support indexing), order needs to be [traces, *additional_data, target] for standard and [traces, *additional_data, target, weights] for weighted
            mask (np.ndarray): The mask to apply to the dataset.
            bs (int, optional): Batch size. Defaults to 32.
            transpose (bool, optional): Wether the traces need to be transposed. Defaults to False.
            traces_mapping (callable, optional): A function which performs a mapping on the traces, should return the traces after transformation. Defaults to None.
            device (str, optional): To which device the data shoulds gets pushed after loading. Defaults to "cpu".
            target_scaling (float, optional): Wether the targets should get scaled, results in target = targets_old * target_scaling. Defaults to 1.
            name (str, optional): Description of the dataloader for easier identificatiom. Defaults to "Undefined".
            weighted (bool, optional): Wether to use weighted training. Defaults to False.
            class_weights (_type_, optional): _description_. Defaults to None.
        """
        # transform batch mask to indices
        self.nonzero_idx, = np.nonzero(mask)
        self.nonzero_idx = self.nonzero_idx.flatten() # make sure that its flat since later used for indexing
        self.transpose = transpose
        self.tensors = tensors
        self.batch_size = bs
        self.traces_mapping = traces_mapping
        self.device = device
        self.target_scaling=target_scaling
        self.name = name
        self.mask = mask
        
        # infer target position from weights since we need to transform target later
        self.weighted = weighted
        self.target_idx = -2 if weighted else -1
        
        if type(class_weights) is np.ndarray:
            self.class_weights = class_weights

    def get_name(self):
        return self.name

    def get_class_weights(self):
        return self.class_weights

    def map_traces(self, traces):
        # transpose dims if necessary
        if self.transpose:
            traces = np.transpose(traces, (0, 2, 1))
        # remove the leads if necessary
        if self.traces_mapping:
            traces = self.traces_mapping(traces)
        return traces

    def map_target(self, target):
        if self.target_scaling != 1:
            target = target * self.target_scaling
        return target

    def get_current_batch(self):
        # We reached the end of the iteration
        if self.start == len(self.nonzero_idx):
            raise StopIteration
        end = min(self.start + self.batch_size, len(self.nonzero_idx))
        batch = [np.array(t[self.nonzero_idx[self.start:end]]) for t in self.tensors]
        self.start = end

        return batch

    def __next__(self):
        # get batch
        batch = self.get_current_batch()

        # map traces
        batch[0] = self.map_traces(batch[0])

        # map target
        batch[self.target_idx] = self.map_target(batch[self.target_idx])

        # convert to torch
        out_value = []
        for b in batch:
            out_value.append(torch.tensor(b.copy(), dtype=torch.float32).to(self.device))
        return out_value

    def __iter__(self):
        self.start = 0 # saves start idx of next batch
        return self

    def __len__(self):
        return math.ceil(len(self.nonzero_idx)/self.batch_size)

    def get_size(self):
        return len(self.nonzero_idx)

    def format_Laplace(self):
        """Formats the dataset for Laplace package by adding the traces to the self.dataset field
        """
        # Laplace libary onky needs to acces loader.dataset to calculate len(loader.dataset), thus we can use the indices to get correct length
        self.dataset = self.nonzero_idx

    def remove_weights(self):
        if len(self.tensors) <= 2:
            print("WARNING: Seems like the dataset already doesn't contain weights. Skipping...")
        else:
            self.tensors = self.tensors[:-1]



class EvenSamplingBatchDataloader(BatchDataloader):
    """A dataloader which samples every batch with even class distribution from the dataset.
    """
    def __init__(self, *data, mask: np.ndarray, bs=32, transpose: bool = False, traces_mapping=None, device: str = "cpu", target_scaling: float = 1, name: str = "Undefined", weighted: bool = False, class_weights=None):
        """
        Args different from parent class BatchDataloader
        Args:
            target_bucketized (_type_): The corresponding class of each target
            bs (int, optional): Make sure that batch size is dividable by number of classes, otherwise batch size will be smaller. Also make sure that there are at least as many examples per class as batch_size/n_classes.
        """
        target_bucketized = data[-1]
        classes = np.unique(target_bucketized) # number of classes
        bs = int(bs/len(classes)) # per class batch size
        super().__init__(*data, mask=mask, bs=bs, transpose=transpose, traces_mapping=traces_mapping, device=device, target_scaling=target_scaling, name=name, weighted=weighted, class_weights=class_weights)
        self.n_datapoints = np.sum(mask)
        self.iterations = int(self.n_datapoints/bs)
        self.class_indices = [np.where(np.logical_and(np.squeeze(target_bucketized) == i, mask))[0] for i in classes]

    def get_current_batch(self):
        if self.current_it >= self.iterations:
            raise StopIteration
        batch_indices = np.sort(np.concatenate([np.random.choice(class_id, self.batch_size, replace=False) for class_id in self.class_indices]))
        batch = [np.array(t[batch_indices]) for t in self.tensors]
        self.current_it += 1
        return batch

    
    def __len__(self):
        return self.iterations

    def get_size(self):
        return self.n_datapoints

    def __iter__(self):
        self.current_it = 0
        return self

    def remove_weights(self):
        pass
