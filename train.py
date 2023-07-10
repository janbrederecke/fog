import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from src.config import TrainGlobalConfig
from src.model import load_model
from src.dataset import FOGDataset
from src.engine import Fitter
from src.folds import get_file_paths

import warnings

warnings.filterwarnings(action="ignore")


def train():
    config = TrainGlobalConfig()

    for fold in range(config.folds):
        model = load_model()

        model = model.to(config.device)

        train_fpaths, valid_fpaths = get_file_paths(
            nfolds=config.folds, fold=fold, config=config
        )

        train_dataset = FOGDataset(
            train_fpaths,
            config=config,
            mode="train",
        )
        valid_dataset = FOGDataset(valid_fpaths, mode="valid", config=config)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=config.batch_size, num_workers=config.num_workers
        )

        fitter = Fitter(model, config.device, config, fold=fold)
        fitter.fit(train_loader, valid_loader)
