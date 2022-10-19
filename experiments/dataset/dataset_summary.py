import h5py
import numpy as np
import argparse
import sys, os
sys.path.insert(0, os.getcwd())
from warnings import warn
import json
import pandas as pd


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
    random_index = np.sort(np.random.choice(f[config_dict["traces_dset"]].shape[0], 1000, replace=False))
    traces = f[config_dict["traces_dset"]][random_index]
    if traces.shape[-1] < traces.shape[-2]:
        traces = np.transpose(traces, (0, 2, 1)) # transpose if wrong order
    df = pd.DataFrame()
    df["leads"] = config_dict["lead_index"]
    # np.set_printoptions(threshold=sys.maxsize)
    df["min"] = np.mean(np.min(traces, axis=-1), axis=0)
    df["max"] = np.mean(np.max(traces, axis=-1), axis=0)
    df["mean"] = np.mean(traces, axis=(0, -1))
    df = df.set_index("leads")
    print(df)
    df.to_csv("Dataset_summary.csv")

if __name__ == "__main__":
    main()
