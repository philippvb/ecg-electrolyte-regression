import argparse
import sys, os
sys.path.insert(0, os.getcwd())
import ecg_plot
import matplotlib.pyplot as plt
from src.evaluations.traces import plot_trace
from src.dataset.ages import get_single_trace
from warnings import warn
import json
import socket
import numpy as np
import h5py
from scipy import signal as sgn

def remove_baseline_filter(sample_rate):
    fc = 0.8  # [Hz], cutoff frequency
    fst = 0.2  # [Hz], rejection band
    rp = 0.5  # [dB], ripple in passband
    rs = 40  # [dB], attenuation in rejection band
    wn = fc / (sample_rate / 2)
    wst = fst / (sample_rate / 2)

    filterorder, aux = sgn.ellipord(wn, wst, rp, rs)
    sos = sgn.iirfilter(filterorder, wn, rp, rs, btype='high', ftype='ellip', output='sos')

    return sos

def get_padding(trace):
    t = trace[0]!=0.0
    print(t)
    lower = np.argmax(t)
    upper = np.argmax(np.flip(t, axis=-1))
    lower = np.median(lower)
    upper = np.median(upper)
    return lower, upper


def main():
    parser = argparse.ArgumentParser(description='Create HDF5 files.')
    parser.add_argument('-f', type=str, help='path to dataset config file')
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    # Get config
    with open(args.f, 'r') as f:
        config_dict = json.load(f)
    f = h5py.File(config_dict["path_to_traces"], 'r')
    random_index = np.sort(np.random.choice(f[config_dict["traces_dset"]].shape[0], 1, replace=False))
    traces = f[config_dict["traces_dset"]][random_index][0].T/2
    # if traces.shape[-1] < traces.shape[-2]:
    #     traces = np.transpose(traces, (0, 2, 1)) # transpose if wrong order

    sos = remove_baseline_filter(400)
    ecg_nobaseline = sgn.sosfiltfilt(sos, traces, padtype='constant', axis=-1)

    print("Finsihed loading")
    # traces = traces.mean(axis=0)
    print("Finished mean")


    plot_trace(traces, config_dict["lead_index"])
    plt.savefig('ecg_plot.png')
    plot_trace(ecg_nobaseline, config_dict["lead_index"])
    plt.savefig('ecg_plot_preprocessed.png')


if __name__ == "__main__":
    main()