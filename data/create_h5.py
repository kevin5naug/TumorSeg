#!/usr/bin/env python

r"""Convert nii to hdf5 file.

Example usage:
    ./create_h5.py --data_dir=/path/to/data_dir_contain_HG_and_LG \
        --output_path=/path/to/h5_file
"""
import os.path as osp
import argparse

import numpy as np
import h5py
import nibabel as nib

# SAMPLE and MODALITY
SAMPLE = ["LG/0001", "LG/0002", "LG/0004", "LG/0006", "LG/0008", "LG/0011",
          "LG/0012", "LG/0013", "LG/0014", "LG/0015", "HG/0001", "HG/0002",
          "HG/0003", "HG/0004", "HG/0005", "HG/0006", "HG/0007", "HG/0008",
          "HG/0009", "HG/0010", "HG/0011", "HG/0012", "HG/0013", "HG/0014",
          "HG/0015", "HG/0022", "HG/0024", "HG/0025", "HG/0026", "HG/0027"]
MODALITY = ["Flair.finalNorm.nii.gz", "T1.finalNorm.nii.gz",
            "T1c.finalNorm.nii.gz", "T2.finalNorm.nii.gz"]

def parse_args():
    """ Parse args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='input_dir',
                        help='dir for training nii data',
                        default='training', type=str)
    parser.add_argument('--output_path', dest='output_path',
                        help='output path for h5 file',
                        default='training.h5', type=str)
    args = parser.parse_args()
    return args

def get_img_data(data_dir, sample, modality):
    """Get image data.
    """
    nii_path = osp.join(data_dir, sample, modality)
    img = nib.load(nii_path)
    return img.get_data()

def main():
    """main loop
    """
    args = parse_args()
    data_dir = args.input_dir
    with h5py.File(args.output_path, 'w') as f:
        for path in SAMPLE:
            img_data = get_img_data(data_dir, path, MODALITY[0])
            dim = img_data.shape
            img_array = np.empty((4, dim[0], dim[1], dim[2]), dtype=np.float32)
            img_array[0, ...] = img_data
            for j in range(1, len(MODALITY)-1):
                img_array[j, ...] = get_img_data(data_dir, path, MODALITY[j])
            f[path] = img_array

if __name__ == '__main__':
    main()
