import random
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from torchvision import transforms


class RandomRotation90:
    def __init__(self, p: float = 0.05) -> None:
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return transforms.functional.rotate(x, 90)
        return x

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p})"


def build_precision_recall_threshold_curves(
    gt_labels: torch.Tensor, probs: torch.Tensor, output_path: Path, class_idx: int
) -> None:
    gt_labels = gt_labels.cpu().numpy()
    probs = probs.cpu().numpy()

    # Convert the multi-class labels to binary labels for the current class
    binary_gt_labels = (gt_labels == class_idx).astype(int)

    precision, recall, thresholds = precision_recall_curve(binary_gt_labels, probs)
    thresholds = np.append(thresholds, 1)  # appending 1 to the end to match the length

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision, label="Precision", color="blue")
    plt.plot(thresholds, recall, label="Recall", color="red")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Precision & Recall vs. Threshold Curve for Class {class_idx}")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def set_seeds(seed: int, cudnn_fixed: bool = False) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if cudnn_fixed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def wandb_logger(loss, metrics: Dict[str, float], mode: str) -> None:
    wandb.log({f"{mode}/loss/": loss})
    for metric_name, metric_value in metrics.items():
        wandb.log({f"{mode}/metrics/{metric_name}": metric_value})


from torch.optim.lr_scheduler import _LRScheduler


class CyclicPlateauLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        start_lr,
        end_lr,
        total_steps,
        steps_up,
        steps_down,
        plateau_steps=1,
        last_epoch=-1,
    ):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.total_steps = total_steps
        self.steps_up = steps_up
        self.steps_down = steps_down
        self.plateau_steps = plateau_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        total_steps = self.steps_up + self.steps_down + self.plateau_steps
        cycle_pos = self.last_epoch % total_steps

        if cycle_pos < self.steps_up:
            # Calculate increasing part
            lr = np.linspace(self.start_lr, self.end_lr, self.steps_up)[cycle_pos]
        elif cycle_pos < self.steps_up + self.steps_down:
            # Calculate decreasing part
            # Exclude the last value of moving_up and the first value of moving_down
            index = cycle_pos - self.steps_up
            lr = np.linspace(self.end_lr, self.start_lr, self.steps_down + 1)[index + 1]
        else:
            # Plateau at the low value
            lr = self.start_lr

        return [lr for _ in self.base_lrs]
