import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_minmax_normalize(values, min_value=None, max_value=None, eps=1e-8):
    values = np.asarray(values, dtype=np.float32).copy()
    if min_value is None:
        min_value = np.min(values, axis=0)
    if max_value is None:
        max_value = np.max(values, axis=0)
    values = (values - min_value) / (max_value - min_value + eps)
    values = np.clip(np.nan_to_num(values), 0.0, 1.0)
    return values.astype(np.float32), min_value, max_value


class SurrogateDataset(Dataset):
    """Normalizes reservoir parameters and production curves for surrogate training."""

    def __init__(
        self,
        x,
        y,
        grid_shape=(1, 60, 60),
        ts_feature=(50, 8),
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        sequence_input=False,
        log_x=False,
    ):
        self.grid_shape = tuple(grid_shape)
        self.ts_feature = tuple(ts_feature)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if log_x:
            x = np.log(np.clip(x, 1e-8, None))

        x, self.x_min, self.x_max = safe_minmax_normalize(x, x_min, x_max)
        x = x.reshape(-1, self.grid_shape[2], self.grid_shape[1], self.grid_shape[0])
        x = np.transpose(x, (0, 3, 2, 1))
        if sequence_input:
            x = x.reshape(-1, self.grid_shape[0], self.grid_shape[1] * self.grid_shape[2])

        y = y.reshape(y.shape[0], self.ts_feature[1], self.ts_feature[0]).swapaxes(1, 2)
        if y_min is None:
            y_min = np.min(y, axis=0)
            for i in range(self.ts_feature[1]):
                y_min[:, i] = y_min[:, i].min()
        if y_max is None:
            y_max = np.max(y, axis=0)
            for i in range(self.ts_feature[1]):
                y_max[:, i] = y_max[:, i].max()
        y, self.y_min, self.y_max = safe_minmax_normalize(y, y_min, y_max)

        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class R2Score(nn.Module):
    def forward(self, prediction, target):
        residual = torch.sum((target - prediction) ** 2)
        total = torch.sum((target - torch.mean(target)) ** 2).clamp_min(1e-8)
        return 1.0 - residual / total


class RMSE(nn.Module):
    def forward(self, prediction, target):
        return torch.sqrt(torch.mean((target - prediction) ** 2))


class TemporalVariationLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = float(weight)

    def forward(self, prediction):
        return self.weight * torch.mean(torch.abs(prediction[:, 1:] - prediction[:, :-1]))


@dataclass
class TrainingHistory:
    train_loss: list
    val_loss: list
    train_r2: list
    val_r2: list
    train_rmse: list
    val_rmse: list
    best_epoch: int
    best_r2: float
    best_rmse: float
    elapsed_seconds: float


def train_surrogate(
    model,
    train_loader,
    val_loader,
    device,
    epochs=200,
    lr=1e-4,
    use_tv_loss=False,
    tv_weight=0.1,
    checkpoint=None,
    use_lr_scheduler=True,
):
    model.to(device)
    criterion = nn.MSELoss()
    tv_loss = TemporalVariationLoss(tv_weight) if use_tv_loss else None
    r2_score = R2Score()
    rmse_score = RMSE()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler = None
    if use_lr_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            threshold=1e-4,
            min_lr=1e-5,
        )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_r2": [],
        "val_r2": [],
        "train_rmse": [],
        "val_rmse": [],
    }
    best_state = None
    best_r2 = -float("inf")
    best_rmse = float("inf")
    best_epoch = 0
    start_time = time.time()

    for epoch in range(int(epochs)):
        train_metrics = _run_epoch(
            model, train_loader, device, criterion, r2_score, rmse_score, optimizer, tv_loss
        )
        val_metrics = _run_epoch(model, val_loader, device, criterion, r2_score, rmse_score)

        if scheduler is not None:
            scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_r2"].append(train_metrics["r2"])
        history["val_r2"].append(val_metrics["r2"])
        history["train_rmse"].append(train_metrics["rmse"])
        history["val_rmse"].append(val_metrics["rmse"])

        if val_metrics["r2"] > best_r2:
            best_r2 = val_metrics["r2"]
            best_rmse = val_metrics["rmse"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_r2={val_metrics['r2']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, optimizer, TrainingHistory(
        train_loss=history["train_loss"],
        val_loss=history["val_loss"],
        train_r2=history["train_r2"],
        val_r2=history["val_r2"],
        train_rmse=history["train_rmse"],
        val_rmse=history["val_rmse"],
        best_epoch=best_epoch,
        best_r2=float(best_r2),
        best_rmse=float(best_rmse),
        elapsed_seconds=time.time() - start_time,
    )


def _run_epoch(model, loader, device, criterion, r2_score, rmse_score, optimizer=None, tv_loss=None):
    is_training = optimizer is not None
    model.train(is_training)
    totals = {"loss": 0.0, "r2": 0.0, "rmse": 0.0, "count": 0}

    iterator = tqdm(loader, leave=False, disable=len(loader) < 2)
    with torch.set_grad_enabled(is_training):
        for x, y in iterator:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            loss = criterion(prediction, y)
            if tv_loss is not None:
                loss = loss + tv_loss(prediction)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = x.size(0)
            totals["loss"] += float(loss.detach()) * batch_size
            totals["r2"] += float(r2_score(prediction.detach(), y)) * batch_size
            totals["rmse"] += float(rmse_score(prediction.detach(), y)) * batch_size
            totals["count"] += batch_size

    count = max(1, totals["count"])
    return {key: totals[key] / count for key in ("loss", "r2", "rmse")}


def save_surrogate_checkpoint(output_dir, model, optimizer, history, model_config=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": asdict(history),
            "model_config": model_config or {},
        },
        output_dir / "model.pth",
    )
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"history": asdict(history), "model_config": model_config or {}}, f, indent=2)


def load_surrogate_checkpoint(path, model, optimizer=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
