import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import DataLoader

from src.config import TrainGlobalConfig
from src.model import load_model
from src.dataset import FOGDataset


def inference(kaggle_models=False):
    config = TrainGlobalConfig()

    test_defog_paths = glob("./data/test/defog/*.csv")
    test_tdcsfog_paths = glob("./data/test/tdcsfog/*.csv")
    test_fpaths = [(f, "de") for f in test_defog_paths] + [
        (f, "tdcs") for f in test_tdcsfog_paths
    ]
    test_dataset = FOGDataset(test_fpaths, mode="inference", config=config)
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, num_workers=config.num_workers
    )

    if kaggle_models:
        weight_array = [
            sorted(glob(f"./weights/FoG/kaggle_model_fold{i}*"))[-1]
            for i in range(config.folds)
        ]
    else:
        weight_array = [
            sorted(glob(f"./weights/FoG/best_mean_average_precision_fold{i}*"))[-1]
            for i in range(config.folds)
        ]

    dfs = [pd.DataFrame({"Id": []}) for _ in range(3)]
    variable_names = ["StartHesitation", "Turn", "Walking"]
    df = pd.DataFrame()

    for weight_id, weight in enumerate(weight_array):
        model = load_model()

        if kaggle_models:
            model.load_state_dict(torch.load(weight, map_location=config.device))
        else:
            model.load_state_dict(
                torch.load(weight, map_location=config.device)["model_state_dict"]
            )

        model = model.to(config.device)
        model.eval()

        n_weights = len(weight_array)
        ids = []
        predictions = []

        print(f"Calculating predictions for model {weight_id} of {n_weights}.")

        for _id, x, _ in tqdm(test_loader):
            with torch.no_grad():
                x = x.to(config.device)
                model_output = model(x)

            ids.extend(_id)
            predictions.extend(list(np.nan_to_num(model_output.cpu().numpy())))

        predictions = np.array(predictions)

        if weight_id == 0:
            for i in range(3):
                dfs[i] = pd.DataFrame({"Id": ids})

        for i, var_name in enumerate(variable_names):
            df_temp = pd.DataFrame(
                {
                    "Id": ids,
                    f"{var_name}_{weight_id}": np.round(predictions[:, i], 5),
                }
            )

            dfs[i] = pd.merge(dfs[i], df_temp, how="left", on="Id").fillna(0.0)

    df["Id"] = ids

    for i, variable_name in enumerate(variable_names):
        df[variable_name] = dfs[i].iloc[:, 1:n_weights].mean(axis=1)

    df.to_csv("./results.csv", index=False)
