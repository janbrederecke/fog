import os
import numpy as np
import sys

from datetime import datetime
import time
import torch

from torch.cuda.amp import GradScaler

from sklearn.metrics import average_precision_score

from glob import glob

"""
The classes in this file were highly influenced or taken from Jan Bremer's
code in https://github.com/JanBrem/AI-NT-proBNP
"""


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Fitter:
    def __init__(self, model, device, config, fold):
        self.config = config
        self.fold = fold
        self.epoch = 0
        self.base_dir = f"./{config.folder}"
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        self.log_path = f"{self.base_dir}/log.txt"
        self.best_summary_loss = 10**5

        self.best_mean_average_precision = 0
        self.device = device
        self.model = model

        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(
            self.optimizer, **config.scheduler_params
        )
        self.scaler = GradScaler()

        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.log(
            f"Fitter prepared. Device is {self.device}. Optimizer is {self.optimizer}."
        )

    def fit(self, train_loader, validation_loader):
        _tr = []
        _val = []
        _mean_average_precision = []
        _average_precision_start_hesitation = []
        _average_precision_turn = []
        _average_precision_walking = []

        for epoch in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]["lr"]
                timestamp = datetime.utcnow().isoformat()
                self.log(f"\n{timestamp}\nLR: {lr}")
            t = time.time()

            summary_loss = self.train_one_epoch(train_loader)
            _tr.append(summary_loss.avg)
            self.log(
                f"[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}"
            )

            t = time.time()

            (
                summary_loss,
                mean_average_precision,
                average_precision_scores,
            ) = self.validation(validation_loader)
            _val.append(summary_loss.avg)
            _mean_average_precision.append(mean_average_precision)
            _average_precision_start_hesitation.append(average_precision_scores[0])
            _average_precision_turn.append(average_precision_scores[1])
            _average_precision_walking.append(average_precision_scores[2])

            self.log(
                f"[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, mean_average_precision {mean_average_precision}, time: {(time.time() - t):.5f}"
            )

            if mean_average_precision > self.best_mean_average_precision:
                self.best_mean_average_precision = mean_average_precision
                self.model.eval()
                self.save(
                    f"{self.base_dir}/best_mean_average_precision_fold{self.fold}_{str(self.epoch).zfill(3)}epoch.bin"
                )
                for path in sorted(
                    glob(
                        f"{self.base_dir}/best_mean_average_precision_fold{self.fold}-*epoch.bin"
                    )
                )[:-3]:
                    os.remove(path)

            if summary_loss.avg < self.best_summary_loss:
                print("saving best model")
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(
                    f"{self.base_dir}/best_checkpoint_fold{self.fold}_{str(self.epoch).zfill(3)}epoch.bin"
                )
                for path in sorted(
                    glob(f"{self.base_dir}/best_checkpoint_fold{self.fold}_*epoch.bin")
                )[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            np.save(
                self.base_dir + f"/log_fold{self.fold}.npy",
                np.array(
                    [
                        _tr,
                        _val,
                        _mean_average_precision,
                        _average_precision_start_hesitation,
                        _average_precision_turn,
                        _average_precision_walking,
                    ]
                ),
            )

            self.epoch += 1

    def validation(self, valid_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        ground_truth = []
        predictions = []
        t_valid_epoch = []

        for step, (sensor_data, outcomes, timepoints) in enumerate(valid_loader):
            sensor_data = sensor_data.to(self.device, dtype=torch.float)
            outcomes = outcomes.to(self.device, dtype=torch.float)
            timepoints = timepoints.to(self.device, dtype=torch.float)

            sys.stdout.write(
                "\r"
                + f"Val Step {step}/{len(valid_loader)}, "
                + f"summary_loss: {summary_loss.avg:.5f}, "
                + f"time: {(time.time() - t):.5f}"
                + "\r"
            )

            with torch.no_grad():
                batch_size = sensor_data.shape[0]
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(sensor_data)
                    assert output.dtype is torch.float16

                predictions.extend(output.detach().cpu().numpy())
                ground_truth.extend(outcomes.detach().cpu().numpy())
                t_valid_epoch.extend(timepoints.detach().cpu().numpy())
                loss = self.loss(output, outcomes)
                loss = torch.mean(loss * timepoints.unsqueeze(-1), dim=1)

                t_sum = torch.sum(timepoints)
                if t_sum > 0:
                    loss = torch.sum(loss) / t_sum
                else:
                    loss = torch.sum(loss) * 0.0

                summary_loss.update(loss.detach().item(), batch_size)

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

        return summary_loss, mean_average_precision, average_precision_scores

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (sensor_data, outcomes, timepoints) in enumerate(train_loader):
            sensor_data = sensor_data.to(self.device, dtype=torch.float)
            outcomes = outcomes.to(self.device, dtype=torch.float)
            timepoints = timepoints.to(self.device, dtype=torch.float)

            sys.stdout.write(
                "\r"
                + f"Train Step {step}/{len(train_loader)}, "
                + f"summary_loss: {summary_loss.avg:.5f}, "
                + f"time: {(time.time() - t):.5f}"
            )

            batch_size = sensor_data.shape[0]

            self.optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = self.model(sensor_data)
                assert output.dtype is torch.float16

            loss = self.loss(output, outcomes)
            loss = torch.mean(loss * timepoints.unsqueeze(-1), dim=1)

            t_sum = torch.sum(timepoints)
            if t_sum > 0:
                loss = torch.sum(loss) / t_sum
            else:
                loss = torch.sum(loss) * 0.0

            self.scaler.scale(loss).backward()

            summary_loss.update(loss.detach().item(), batch_size)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.config.step_scheduler:
                self.scheduler.step()
        print("")
        return summary_loss

    def save(self, path):
        self.model.eval()
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_summary_loss": self.best_summary_loss,
                "epoch": self.epoch,
            },
            path,
        )

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, "a+") as logger:
            logger.write(f"{message}\n")
