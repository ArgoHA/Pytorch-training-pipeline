import random
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig
from PIL import Image, ImageOps
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from src.utils import RandomRotation90, get_class_names


class CustomDataset(Dataset):
    def __init__(
        self,
        root_path: Path,
        split: pd.DataFrame,
        train_mode: bool,
    ):
        self.root_path = root_path
        self.split = split
        self.img_size = (256, 256)
        self._init_augs(train_mode)

    def _init_augs(self, train_mode: bool) -> None:
        if train_mode:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    RandomRotation90(p=0.05),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    def __getitem__(self, idx: int):
        image_path, label = self.split.iloc[idx]

        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)  # fix rotation
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.split)


class Loader:
    def __init__(self, root_path: Path, batch_size: int, num_workers: int):
        self.root_path = root_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.get_splits()

    def get_splits(self) -> None:
        self.splits = {"train": None, "val": None, "test": None}
        for split_name in self.splits.keys():
            if (self.root_path / f"{split_name}.csv").exists():
                self.splits[split_name] = pd.read_csv(
                    self.root_path / f"{split_name}.csv", header=None
                )
            else:
                self.splits[split_name] = []

    def build_dataloaders(self) -> None:
        # build datasets
        train_ds = CustomDataset(self.root_path, self.splits["train"], train_mode=True)
        val_ds = CustomDataset(self.root_path, self.splits["val"], train_mode=False)

        if len(self.splits["test"]):
            test_ds = CustomDataset(
                self.root_path, self.splits["test"], train_mode=False
            )

        # build dataloaders
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        if len(self.splits["test"]):
            test_loader = DataLoader(
                test_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
        else:
            test_loader = None

        return train_loader, val_loader, test_loader

    @property
    def num_classes(self) -> int:
        return len(get_class_names(self.root_path))


def prepare_model(
    model_path: Path, num_classes: int, device: str, model_name: str
) -> nn.Module:
    model = build_model(num_classes, device, model_name, layers_to_train=-1)
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def build_model(
    n_outputs: int, device: str, model_name: str, layers_to_train: int
) -> nn.Module:
    if model_name == "shuffle_net":
        model = models.shufflenet_v2_x0_5(
            weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT
        )
        model.fc = nn.Linear(in_features=1024, out_features=n_outputs)
    elif model_name == "eficient_net_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=n_outputs)
    else:
        raise Exception("Model not found")

    if layers_to_train == -1:
        return model.to(device)

    for param in list(model.parameters())[:-layers_to_train]:
        param.requires_grad = False

    return model.to(device)


def get_preds(
    model: nn.Module, val_loader: DataLoader, device: str
) -> Tuple[List[int]]:
    val_preds = []
    val_labels = []
    model.eval()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)

            val_preds.extend(torch.max(outputs.data, 1).indices.tolist())
            val_labels.extend(labels.tolist())

    return val_preds, val_labels


def wandb_logger(loss, metrics: Dict[str, float], mode: str) -> None:
    wandb.log({"loss": loss})
    for metric_name, metric_value in metrics.items():
        wandb.log({f"{mode}/metrics/{metric_name}": metric_value})


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_metrics(gt_labels: List[int], preds: List[int]) -> Dict[str, float]:
    metrics = {}
    metrics["accuracy"] = accuracy_score(gt_labels, preds)
    metrics["f1"] = f1_score(gt_labels, preds)
    metrics["precision"] = precision_score(gt_labels, preds)
    metrics["recall"] = recall_score(gt_labels, preds)
    return metrics


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    model: nn.Module,
    loss_func: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    epochs: int,
    path_to_save: Path,
) -> None:

    best_metric = 0
    wandb.watch(model, log_freq=100)
    for epoch in range(1, epochs + 1):
        model.train()

        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)
                tepoch.set_description(f"Epoch {epoch}/{epochs}")

                optimizer.zero_grad()

                # with torch.cuda.amp.autocast():
                output = model(inputs)
                loss = loss_func(output, labels)

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

        preds, gt_labels = get_preds(model, val_loader, device)
        metrics = get_metrics(gt_labels, preds)

        scheduler.step()

        print(
            f"Val f1: {round(metrics['f1'], 3)}, Val accuracy: {round(metrics['accuracy'], 3)}"
        )

        if metrics["f1"] > best_metric:
            best_metric = metrics["f1"]

            print("Saving new best model...")
            path_to_save.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path_to_save)

        wandb_logger(loss, metrics, mode="val")


def evaluate(test_loader: DataLoader, model: nn.Module, device: str, mode: str) -> None:
    preds, gt_labels = get_preds(model, test_loader, device)
    metrics = get_metrics(gt_labels, preds)
    print(
        f"{mode.capitalize()} f1: {round(metrics['f1'], 3)}, {mode.capitalize()} accuracy: {round(metrics['accuracy'], 3)}"
    )

    wandb_logger(None, metrics, mode=mode)


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.train.seed)
    wandb.init(
        project=cfg.project_name,
        name=cfg.exp,
        config=cfg,
    )
    cfg = cfg.train

    base_loader = Loader(Path(cfg.data_path), cfg.batch_size, cfg.num_workers)
    train_loader, val_loader, test_loader = base_loader.build_dataloaders()

    model = build_model(
        base_loader.num_classes, cfg.device, cfg.model_name, cfg.layers_to_train
    )

    loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

    try:
        train(
            train_loader=train_loader,
            val_loader=val_loader,
            device=cfg.device,
            model=model,
            loss_func=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=cfg.epochs,
            path_to_save=Path(cfg.path_to_save) / "model.pt",
        )

    finally:
        model = prepare_model(
            Path(cfg.path_to_save) / "model.pt",
            base_loader.num_classes,
            cfg.device,
            cfg.model_name,
        )

        evaluate(val_loader, model, cfg.device, mode="val")

        if test_loader is not None:
            evaluate(test_loader, model, cfg.device, mode="test")


if __name__ == "__main__":
    main()

"""
TODO:
- add early stopping

"""
