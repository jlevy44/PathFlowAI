# import brambox as bb
# import os
from os.path import join

# from os.path import basename
from pathflowai.utils import load_sql_df, npy2da, df2sql

# import skimage
import dask

# import dask.array as da
import pandas as pd
import numpy as np
import argparse

# from scipy import ndimage
from scipy.ndimage.measurements import label

# import pickle
# from dask.distributed import Client
from multiprocessing import Pool
from functools import reduce


def count_cells(m, num_classes=3):
    lbls, n_lbl = label(m)
    obj_labels = np.zeros(num_classes)
    for i in range(1, num_classes + 1):
        obj_labels[i - 1] = len(np.unique(lbls[m == i].flatten()))
    return obj_labels


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_classes", default=4, type=int)
    p.add_argument("--patch_size", default=512, type=int)
    p.add_argument("--n_workers", default=40, type=int)
    p.add_argument("--p_sample", default=0.7, type=float)
    p.add_argument("--input_dir", default="inputs", type=str)
    p.add_argument("--patch_info_file", default="cell_info.db", type=str)
    p.add_argument("--reference_mask", default="reference_mask.npy", type=str)
    # c=Client()
    # add mode to just use own extracted boudning boxes or from seg, maybe from histomicstk

    args = p.parse_args()
    num_classes = args.num_classes
    n_workers = args.n_workers
    input_dir = args.input_dir
    patch_info_file = args.patch_info_file
    patch_size = args.patch_size
    np.random.seed(42)
    reference_mask = args.reference_mask

    patch_info = load_sql_df(patch_info_file, patch_size)
    IDs = patch_info["ID"].unique()
    # slides = {slide:da.from_zarr(join(input_dir,'{}.zarr'.format(slide))) for slide in IDs}
    masks = {mask: npy2da(join(input_dir, "{}_mask.npy".format(mask))) for mask in IDs}

    def process_chunk(patch_info_sub):
        patch_info_sub = patch_info_sub.reset_index(drop=True)
        counts = []
        for i in range(patch_info_sub.shape[0]):
            # print(i)
            patch = patch_info_sub.iloc[i]
            ID, x, y, patch_size2 = patch[["ID", "x", "y", "patch_size"]].tolist()
            m = masks[ID][x : x + patch_size2, y : y + patch_size2]
            counts.append(dask.delayed(count_cells)(m, num_classes=num_classes))

        return dask.compute(*counts, scheduler="threading")

    patch_info_subs = np.array_split(patch_info, n_workers)

    p = Pool(n_workers)

    counts = reduce(lambda x, y: x + y, p.map(process_chunk, patch_info_subs))

    # bbox_dfs=dask.compute(*bbox_dfs,scheduler='processes')

    counts = pd.DataFrame(np.vstack(counts))

    patch_info = pd.concat(
        [
            patch_info[["ID", "x", "y", "patch_size", "annotation"]].reset_index(
                drop=True
            ),
            counts.reset_index(drop=True),
        ],
        axis=1,
    ).reset_index()
    print(patch_info)

    df2sql(patch_info, "counts_test.db", patch_size, mode="replace")
