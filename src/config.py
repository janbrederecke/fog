import torch


class TrainGlobalConfig:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    train_dir_tdcsfog = "./data/train/tdcsfog/"
    train_dir_defog = "./data/train/defog/"

    test_dir = "./data/test/"

    metadata_tdcsfog = "./data/tdcsfog_metadata.csv"
    metadata_defog = "./data/defog_metadata.csv"

    window_size = 1000
    window_future = 50
    window_past = window_size - window_future

    folds = 5
    num_workers = 4
    batch_size = 1024
    n_epochs = 1
    lr = 0.001

    folder = "weights/FoG/"
    verbose = True
    verbose_step = 1

    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode="min", factor=0.7, patience=10, verbose=False, min_lr=0.000001
    )
