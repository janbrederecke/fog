import numpy as np
import pandas as pd

import random
import gc
import os
import time

import torch
from torch.utils.data import Dataset

"""
This dataset class was highly influenced by Mayukh Bhattacharyya's public
notebook posted in the Kaggle discussion forum for the FoG-competition:
https://www.kaggle.com/code/mayukh18/pytorch-fog-end-to-end-baseline-lb-0-254
"""


class FOGDataset(Dataset):
    def __init__(self, file_paths, config, mode="train", verbose=True):
        super().__init__()
        tm = time.time()
        self.mode = mode
        if self.mode not in ["train", "test", "inference", "valid"]:
            raise ValueError(
                '{self.mode} must be "train", "test"/"inference", or "valid"'
            )
        self.config = config

        self.file_paths = file_paths
        self.dfs = [self.read(f[0], f[1]) for f in file_paths]

        self.f_ids = [os.path.basename(f[0])[:-4] for f in self.file_paths]

        self.end_indices = []
        self.shapes = []
        _length = 0
        for df in self.dfs:
            self.shapes.append(df.shape[0])
            _length += df.shape[0]
            self.end_indices.append(_length)

        self.dfs = np.concatenate(self.dfs, axis=0).astype(np.float16)
        self.length = self.dfs.shape[0]

        shape1 = self.dfs.shape[1]

        self.dfs = np.concatenate(
            [
                np.zeros((self.config.window_past, shape1)),
                self.dfs,
                np.zeros((self.config.window_future, shape1)),
            ],
            axis=0,
        )
        if verbose:
            print(f"Dataset initialized in {time.time() - tm} secs!")
        gc.collect()

    def read(self, f, _type):
        df = pd.read_csv(f)
        if self.mode == "test":
            return np.array(df)

        if _type == "tdcs":
            df["Valid"] = 1
            df["Task"] = 1
            df["tdcs"] = 1
        else:
            df["tdcs"] = 0

        return np.array(df)

    def __getitem__(self, index):
        if self.mode == "train":
            row_idx = random.randint(0, self.length - 1) + self.config.window_past
        elif self.mode in ["test", "inference"]:
            for i, e in enumerate(self.end_indices):
                if index >= e:
                    continue
                df_idx = i
                break

            row_idx_true = self.shapes[df_idx] - (self.end_indices[df_idx] - index)
            _id = self.f_ids[df_idx] + "_" + str(row_idx_true)
            row_idx = index + self.config.window_past
        else:
            row_idx = index + self.config.window_past

        x = self.dfs[
            row_idx - self.config.window_past : row_idx + self.config.window_future, 1:4
        ]

        x = x[::-1, :]  # This line is only necessary to use the original models

        x = torch.tensor(x.astype("float32")).T

        t = self.dfs[row_idx, -3] * self.dfs[row_idx, -2]
        t = torch.tensor(t).half()

        if self.mode in ["inference", "test"]:
            return _id, x, t

        y = self.dfs[row_idx, 4:7].astype("float32")
        y = torch.tensor(y)

        return x, y, t

    def __len__(self):
        if self.mode == "train":
            return 5_000_000
        return self.length
