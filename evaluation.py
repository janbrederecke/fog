import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from glob import glob
from tqdm import tqdm

from src.config import TrainGlobalConfig
from src.model import load_model
from src.dataset import FOGDataset
from src.folds import get_file_paths

import warnings

warnings.filterwarnings(action="ignore")


def evaluation(kaggle_models=False):
    config = TrainGlobalConfig()

    assert (config.device is "cuda", "Device has to be CUDA for this operation.")

    if kaggle_models:
        weight_array = [
            sorted(glob(f"./weights/FoG/kaggle_model_fold{i}*"))[-1]
            for i in range(config.folds)
        ]
        model_type = "kaggle_models"
    else:
        weight_array = [
            sorted(glob(f"./weights/FoG/best_mean_average_precision_fold{i}*"))[-1]
            for i in range(config.folds)
        ]
        model_type = "new_models"

    _mean_average_precision = []
    _average_precision_start_hesitation = []
    _average_precision_turn = []
    _average_precision_walking = []

    for fold in range(config.folds):
        model = load_model()

        if kaggle_models:
            model.load_state_dict(
                torch.load(weight_array[fold], map_location=config.device)
            )
        else:
            model.load_state_dict(
                torch.load(weight_array[fold], map_location=config.device)[
                    "model_state_dict"
                ]
            )

        model = model.to(config.device)
        model.eval()

        _, valid_fpaths = get_file_paths(
            nfolds=config.folds, fold=fold, config=config, verbose=False
        )

        valid_dataset = FOGDataset(
            valid_fpaths, mode="valid", config=config, verbose=False
        )

        valid_loader = DataLoader(
            valid_dataset, batch_size=config.batch_size, num_workers=config.num_workers
        )
        ground_truth = []
        predictions = []
        t_valid_epoch = []

        for step, (sensor_data, outcomes, timepoints) in enumerate(tqdm(valid_loader)):
            sensor_data = sensor_data.to(config.device, dtype=torch.float)
            outcomes = outcomes.to(config.device, dtype=torch.float)
            timepoints = timepoints.to(config.device, dtype=torch.float)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(sensor_data)
                    assert output.dtype is torch.float16

                predictions.extend(output.detach().cpu().numpy())
                ground_truth.extend(outcomes.detach().cpu().numpy())
                t_valid_epoch.extend(timepoints.detach().cpu().numpy())

        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        t_valid_epoch = np.array(t_valid_epoch)

        predictions = predictions[t_valid_epoch > 0, :]
        ground_truth = ground_truth[t_valid_epoch > 0, :]

        average_precision_scores = [
            average_precision_score(ground_truth[:, i], predictions[:, i])
            for i in range(3)
        ]
        mean_average_precision = np.mean(average_precision_scores)

        _mean_average_precision.append(mean_average_precision)
        _average_precision_start_hesitation.append(average_precision_scores[0])
        _average_precision_turn.append(average_precision_scores[1])
        _average_precision_walking.append(average_precision_scores[2])

    results = pd.DataFrame(
        {
            "fold": [i for i in range(config.folds)],
            "map": _mean_average_precision,
            "ap_start_hesitation": _average_precision_start_hesitation,
            "ap_turn": _average_precision_turn,
            "ap_walking": _average_precision_walking,
        }
    )

    np.save(
        config.folder + f"/evaluation_metrics_{model_type}.npy",
        results,
    )

    print(results)
