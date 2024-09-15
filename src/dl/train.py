from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import timm
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import autocast, nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from src.dl.utils import (
    RandomRotation90,
    build_precision_recall_threshold_curves,
    set_seeds,
    wandb_logger,
)
from src.ptypes import class_names, img_norms, label_to_name_mapping


class CustomDataset(Dataset):
    def __init__(
        self,
        img_size: Tuple[int, int],
        root_path: Path,
        split: pd.DataFrame,
        debug_img_processing: bool,
        train_mode: bool,
    ) -> None:
        self.root_path = root_path
        self.split = split
        self.img_size = img_size
        self.norm = img_norms
        self.debug_img_processing = debug_img_processing
        self._init_augs(train_mode)

    def _init_augs(self, train_mode: bool) -> None:
        if train_mode:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.Lambda(self._convert_rgb),
                    RandomRotation90(p=0.2),
                    transforms.RandomHorizontalFlip(p=0.1),
                    transforms.RandomVerticalFlip(p=0.1),
                    # transforms.RandAugment(num_ops=2, magnitude=10),
                    transforms.ToTensor(),
                    transforms.Normalize(*self.norm),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.Lambda(self._convert_rgb),
                    transforms.ToTensor(),
                    transforms.Normalize(*self.norm),
                ]
            )

    def _convert_rgb(self, x: Image) -> Image:
        return x.convert("RGB")

    def _save_transformed_images(
        self, image: torch.Tensor, idx: int, label: int, image_path: str
    ) -> None:
        def unnormalize(tensor: torch.Tensor):
            for t, m, s in zip(tensor, self.norm[0], self.norm[1]):
                t.mul_(s).add_(m)
            return tensor

        save_path = self.root_path.parent / "output" / "debug_img_processing"
        save_path.mkdir(exist_ok=True, parents=True)

        unnorm_img_tensor = unnormalize(image.clone())
        transforms.ToPILImage()(unnorm_img_tensor).save(
            save_path / f"{Path(image_path).stem}_{idx}_{label}.jpg"
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.split.iloc[idx]

        image = Image.open(self.root_path / image_path)
        image.draft("RGB", self.img_size)  # speeds up loading
        image = ImageOps.exif_transpose(image)  # fix rotation
        image = self.transform(image)

        if self.debug_img_processing:
            self._save_transformed_images(image, idx, label, image_path)
        return image, label

    def __len__(self) -> int:
        return len(self.split)


class Loader:
    def __init__(
        self,
        root_path: Path,
        img_size: Tuple[int, int],
        batch_size: int,
        num_workers: int,
        debug_img_processing: bool = False,
    ) -> None:
        self.root_path = root_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug_img_processing = debug_img_processing
        self._get_splits()
        self.class_names = class_names
        self.print_class_distribution()

    def _get_splits(self) -> None:
        self.splits = {"train": None, "val": None, "test": None}
        for split_name in self.splits.keys():
            if (self.root_path / f"{split_name}.csv").exists():
                self.splits[split_name] = pd.read_csv(
                    self.root_path / f"{split_name}.csv", header=None
                )
            else:
                self.splits[split_name] = []

    def print_class_distribution(self) -> None:
        all_data = pd.concat([split for split in self.splits.values() if np.any(split)])
        class_counts = all_data[1].value_counts().sort_index()
        class_distribution = {}
        for class_id, count in class_counts.items():
            class_distribution[label_to_name_mapping[class_id]] = count
        logger.info(", ".join(f"{key}: {value}" for key, value in class_distribution.items()))

    def _build_dataloader_impl(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )
        dataloader.num_classes = self.num_classes
        dataloader.class_names = self.class_names
        return dataloader

    def build_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_ds = CustomDataset(
            self.img_size,
            self.root_path,
            self.splits["train"],
            self.debug_img_processing,
            train_mode=True,
        )
        val_ds = CustomDataset(
            self.img_size,
            self.root_path,
            self.splits["val"],
            self.debug_img_processing,
            train_mode=False,
        )

        train_loader = self._build_dataloader_impl(train_ds, shuffle=True)
        val_loader = self._build_dataloader_impl(val_ds)

        test_loader = None
        test_ds = []
        if len(self.splits["test"]):
            test_ds = CustomDataset(
                self.img_size,
                self.root_path,
                self.splits["test"],
                self.debug_img_processing,
                train_mode=False,
            )
            test_loader = self._build_dataloader_impl(test_ds)

        logger.info(f"train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
        return train_loader, val_loader, test_loader

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def build_model(n_outputs: int, device: str, layers_to_train: int) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1280, out_features=n_outputs),
    )

    if layers_to_train == -1:
        return model.to(device)
    for param in list(model.parameters())[:-layers_to_train]:
        param.requires_grad = False
    return model.to(device)


def prepare_model(model_path: Path, num_classes: int, device: str) -> nn.Module:
    model = build_model(num_classes, device, layers_to_train=-1)
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def get_metrics(gt_labels: List[int], preds: List[int]) -> Dict[str, float]:
    num_classes = len(set(gt_labels))
    if num_classes == 2:
        average = "binary"
    else:
        average = "macro"

    metrics = {}
    metrics["accuracy"] = accuracy_score(gt_labels, preds)
    metrics["f1"] = f1_score(gt_labels, preds, average=average)
    metrics["precision"] = precision_score(gt_labels, preds, average=average)
    metrics["recall"] = recall_score(gt_labels, preds, average=average)
    return metrics


def postprocess(probs: torch.Tensor, gt_labels: torch.Tensor) -> Tuple[List[int], List[int]]:
    preds = torch.argmax(probs, dim=1).tolist()
    gt_labels = gt_labels.tolist()
    return preds, gt_labels


def evaluate(
    test_loader: DataLoader,
    model: nn.Module,
    device: str,
    path_to_save: Path,
    mode: str,
) -> Dict[str, float]:
    probs, gt_labels = get_full_preds(model, test_loader, device)

    if path_to_save is not None:
        for class_idx in range(test_loader.num_classes):
            output_path = path_to_save / "pr_curves"
            output_path.mkdir(exist_ok=True)

            build_precision_recall_threshold_curves(
                gt_labels,
                probs[:, class_idx],
                output_path / f"{mode}_pr_curve_class_{test_loader.class_names[class_idx]}.png",
                class_idx,
            )

    preds, gt_labels = postprocess(probs, gt_labels)
    metrics = get_metrics(gt_labels, preds)
    return metrics


def get_full_preds(
    model: nn.Module, val_loader: DataLoader, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    val_probs = []  # List to store predicted probabilities for all classes
    val_labels = []
    model.eval()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model.forward(inputs)
            probs = torch.softmax(logits, dim=1)

            val_probs.append(probs)
            val_labels.extend(labels)

    val_probs = torch.cat(val_probs, dim=0)
    val_labels = torch.tensor(val_labels)
    return val_probs, val_labels


def log_metrics_locally(all_metrics: Dict[str, Dict[str, float]], path_to_save: Path) -> None:
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")
    metrics_df = metrics_df.round(4)
    if path_to_save:
        metrics_df.to_csv(path_to_save / "metrics.csv")
    print(metrics_df, "\n")


def save_metrics(train_metrics, metrics, loss, path_to_save) -> None:
    log_metrics_locally(
        all_metrics={"train": train_metrics, "val": metrics}, path_to_save=path_to_save
    )
    wandb_logger(loss, train_metrics, mode="train")
    wandb_logger(None, metrics, mode="val")


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
    scaler = GradScaler()

    for epoch in range(1, epochs + 1):
        model.train()

        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)
                tepoch.set_description(f"Epoch {epoch}/{epochs}")

                optimizer.zero_grad()

                with autocast(device_type=device, dtype=torch.float16):
                    output = model(inputs)
                    loss = loss_func(output, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                tepoch.set_postfix(loss=loss.item())

        train_metrics = evaluate(
            test_loader=train_loader, model=model, device=device, path_to_save=None, mode="train"
        )
        metrics = evaluate(
            test_loader=val_loader, model=model, device=device, path_to_save=None, mode="val"
        )

        if scheduler:
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})
            scheduler.step()

        if metrics["f1"] > best_metric:
            best_metric = metrics["f1"]

            logger.info("Saving new best model...")
            path_to_save.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path_to_save)

        save_metrics(train_metrics, metrics, loss, path_to_save=None)


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.train.seed, cfg.train.cudnn_fixed)
    wandb.init(
        project=cfg.project_name,
        name=cfg.exp,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    cfg = cfg.train

    base_loader = Loader(
        root_path=Path(cfg.data_path),
        img_size=tuple(cfg.img_size),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        debug_img_processing=cfg.debug_img_processing,
    )
    train_loader, val_loader, test_loader = base_loader.build_dataloaders()

    model = build_model(base_loader.num_classes, cfg.device, cfg.layers_to_train)

    loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    scheduler = None
    if cfg["use_scheduler"]:
        scheduler = CyclicLR(
            optimizer,
            base_lr=0.0001,
            max_lr=0.001,
            step_size_up=cfg["epochs"] // 4,
            step_size_down=cfg["epochs"] // 2 - cfg["epochs"] % 4,
            cycle_momentum=False,
        )

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

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(e)
    finally:
        logger.info("Evaluating best model...")
        model = prepare_model(
            model_path=Path(cfg.path_to_save) / "model.pt",
            num_classes=base_loader.num_classes,
            device=cfg.device,
        )

        val_metrics = evaluate(
            test_loader=val_loader,
            model=model,
            device=cfg.device,
            path_to_save=Path(cfg.path_to_save),
            mode="val",
        )

        test_metrics = {}
        if test_loader:
            test_metrics = evaluate(
                test_loader=test_loader,
                model=model,
                device=cfg.device,
                path_to_save=Path(cfg.path_to_save),
                mode="test",
            )
            wandb_logger(None, test_metrics, mode="test")

        log_metrics_locally(
            all_metrics={"val": val_metrics, "test": test_metrics},
            path_to_save=Path(cfg.path_to_save),
        )


if __name__ == "__main__":
    main()
