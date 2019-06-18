#!/usr/bin/env python3

"""
Averaging NPZ files
"""

import argparse
import os
import numpy as np


def get_last_checkpoints(folder, num_checkpoints):
    labels = []
    for fname in os.listdir(folder):
        if not fname.startswith('model-') or not fname.endswith('.npz'):
            continue
        label = fname[len('model-'):-len('.npz')]
        if not label.isdigit():
            continue
        labels.append(int(label))
    labels = sorted(labels, reverse=True)[0:num_checkpoints]

    files = []
    for label in labels:
        filename = os.path.join(folder, 'model-%d.npz' % label)
        files += [filename]
    return files


def average_npzs(files):
    out = {}
    for filename in files:
        model = np.load(filename)
        for var in model:
            if var in out:
                out[var] += model[var]
            else:
                out[var] = model[var]
    for var in out:
        out[var] /= len(files)

    return out


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--oname','-O', required=True, help='output file name')
    p.add_argument('--ncheckpoints', '-n', type=int, help='number of checkpoints to use')
    p.add_argument('--folder', type=str, help='path to checkpoints')
    p.add_argument('files', nargs='*')

    args = p.parse_args()
    if (args.folder is None) != (args.ncheckpoints is None):
        raise Exception("--folder and --ncheckpoints should be specified togather")

    if (args.folder is not None) and len(args.files):
        raise Exception("Use one of two modes:\n<SCRIPT> -O <out_file> files+\n or <SCRIPT> -O <out_file> -n <ncheckpoints> --folder <folder>")

    return args


if __name__ == '__main__':
    args = _parse_args()

    if args.folder:
        files = get_last_checkpoints(args.folder, args.ncheckpoints)
        npz = average_npzs(files)
        np.savez(args.oname, **npz)
    else:
        npz = average_npzs(args.files)
        np.savez(args.oname, **npz)
