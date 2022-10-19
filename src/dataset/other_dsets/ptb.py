import numpy as np
import wfdb
from scipy import signal as sgn
import torch
from src.dataset.utils import compute_weights, remove_baseline_filter


BASELINE_FILTER_400 = remove_baseline_filter(400)


def map_ptb_to_swedish(traces, new_length=4096):
    # we have the leads in the following order:
    # DI, DII, DIII, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
    # and we want to drop III, aVR, aVL, aVF, so drop cols 3-5
    indices = np.array([0, 1, 6, 7, 8, 9, 10, 11])
    traces = np.take(traces, indices, axis=1)
    # resample and filter
    traces = sgn.resample_poly(traces, up=400, down=500, axis=-1)
    traces = sgn.sosfiltfilt(BASELINE_FILTER_400, traces, padtype='constant', axis=-1)
    # pad
    orig_length = traces.shape[-1]
    traces_new = np.zeros((traces.shape[0], traces.shape[1], new_length))
    pad = (new_length - orig_length) // 2
    traces_new[...,pad:pad + orig_length] = traces
    return traces_new



class PTBBatchDataloader:
    def __init__(self, metadatafr, traces_path, traces_path_id="filename_hr", ages_id="age", bs=32, mask=None, transpose=True, traces_mapping=map_ptb_to_swedish, device="cpu"):
        nonzero_idx, = np.nonzero(mask)
        self.traces_path = traces_path
        self.traces_path_id = traces_path_id
        self.ages_id = ages_id
        self.transpose = transpose
        self.metadatafr = metadatafr
        self.batch_size = bs
        self.mask = mask
        self.traces_mapping = traces_mapping
        self.device = device
        if nonzero_idx.size > 0:
            self.start_idx = min(nonzero_idx)
            self.end_idx = max(nonzero_idx)
        else:
            self.start_idx = 0
            self.end_idx = 0

    def __next__(self):

        # compute current start, stop and batch mask
        if self.start == self.end_idx:
            raise StopIteration
        end = min(self.start + self.batch_size, self.end_idx)
        batch_mask = self.mask[self.start:end]
        while sum(batch_mask) == 0:
            self.start = end
            end = min(self.start + self.batch_size, self.end_idx)
            batch_mask = self.mask[self.start:end]
        
        # extract the traces + ages
        traces = np.array([wfdb.rdsamp(self.traces_path+filename)[0] for filename in self.metadatafr[self.start:end][self.traces_path_id]])
        ages = np.array(self.metadatafr[self.start:end][self.ages_id])
        weights = compute_weights(ages)


        batch_mask = np.array(batch_mask, dtype=bool)

        # update pointers for next epoch
        self.start = end
        self.sum += sum(batch_mask)

        # transpose dims if necessary
        if self.transpose:
            traces = np.transpose(traces, (0, 2, 1))
        # remove the leads if necessary
        if self.traces_mapping:
            traces = self.traces_mapping(traces)


        out_value = [torch.tensor(b[batch_mask], dtype=torch.float32).to(self.device) for b in [traces, ages, weights]]
        return out_value

    def __iter__(self):
        self.start = self.start_idx
        self.sum = 0
        return self

    def __len__(self):
        count = 0
        start = self.start_idx
        while start != self.end_idx:
            end = min(start + self.batch_size, self.end_idx)
            batch_mask = self.mask[start:end]
            if sum(batch_mask) != 0:
                count += 1
            start = end
        return count

    def get_size(self):
        return sum(self.mask)
