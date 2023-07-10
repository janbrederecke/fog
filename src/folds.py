import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedGroupKFold


def get_file_paths(nfolds, fold, config, verbose=True):
    metadata_tdcsfog = pd.read_csv(config.metadata_tdcsfog)
    metadata_defog = pd.read_csv(config.metadata_defog)

    sgkf = StratifiedGroupKFold(n_splits=nfolds, random_state=42, shuffle=True)

    # tdcsfog
    for i, (train_index, valid_index) in enumerate(
        sgkf.split(
            X=metadata_tdcsfog["Id"],
            y=[1] * len(metadata_tdcsfog),
            groups=metadata_tdcsfog["Subject"],
        )
    ):
        if i != fold:
            continue
        if verbose:
            print(f"Fold = {i}")
        train_ids = metadata_tdcsfog.loc[train_index, "Id"]
        valid_ids = metadata_tdcsfog.loc[valid_index, "Id"]
        if verbose:
            print(
                f"Length of Train(tdcsfog) = {len(train_ids)}, Length of Valid(tdcsfog) = {len(valid_ids)}"
            )

        if i == fold:
            break

    train_fpaths_tdcs = [f"{config.train_dir_tdcsfog}{_id}.csv" for _id in train_ids]
    valid_fpaths_tdcs = [f"{config.train_dir_tdcsfog}{_id}.csv" for _id in valid_ids]

    # defog
    # Remove entries with no events first
    metadata_defog["n1_sum"] = 0
    metadata_defog["n2_sum"] = 0
    metadata_defog["n3_sum"] = 0
    metadata_defog["count"] = 0

    for f in metadata_defog["Id"]:
        fpath = f"{config.train_dir_defog}{f}.csv"
        if os.path.exists(fpath) == False:
            continue

        df = pd.read_csv(fpath)
        metadata_defog.loc[metadata_defog["Id"] == f, "n1_sum"] = np.sum(
            df["StartHesitation"]
        )
        metadata_defog.loc[metadata_defog["Id"] == f, "n2_sum"] = np.sum(df["Turn"])
        metadata_defog.loc[metadata_defog["Id"] == f, "n3_sum"] = np.sum(df["Walking"])
        metadata_defog.loc[metadata_defog["Id"] == f, "count"] = len(df)

    metadata_defog = metadata_defog[metadata_defog["count"] > 0].reset_index()

    for i, (train_index, valid_index) in enumerate(
        sgkf.split(
            X=metadata_defog["Id"],
            y=[1] * len(metadata_defog),
            groups=metadata_defog["Subject"],
        )
    ):
        if i != fold:
            continue

        train_ids = metadata_defog.loc[train_index, "Id"]
        valid_ids = metadata_defog.loc[valid_index, "Id"]
        if verbose:
            print(
                f"Length of Train(defog) = {len(train_ids)}, Length of Valid(defog) = {len(valid_ids)}"
            )

        if i == fold:
            break

    train_fpaths_de = [f"{config.train_dir_defog}{_id}.csv" for _id in train_ids]
    valid_fpaths_de = [f"{config.train_dir_defog}{_id}.csv" for _id in valid_ids]

    # Combine tdcsfog and defog for output
    train_fpaths = [(f, "de") for f in train_fpaths_de] + [
        (f, "tdcs") for f in train_fpaths_tdcs
    ]
    valid_fpaths = [(f, "de") for f in valid_fpaths_de] + [
        (f, "tdcs") for f in valid_fpaths_tdcs
    ]

    if verbose:
        print(
            f"Length of combined Train = {len(train_fpaths)}, Length of combined Valid = {len(valid_fpaths)}"
        )

    # Return train and valid file paths for given fold
    return train_fpaths, valid_fpaths
