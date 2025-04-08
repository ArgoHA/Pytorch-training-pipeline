import random
import subprocess
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import torch
import wandb
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from tabulate import tabulate

from src.ptypes import label_to_name_mapping


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


def seed_worker(worker_id):  # noqa
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def wandb_logger(loss, metrics: Dict[str, float], epoch, mode: str) -> None:
    log_data = {"epoch": epoch}
    if loss:
        log_data[f"{mode}/loss/"] = loss

    for metric_name, metric_value in metrics.items():
        log_data[f"{mode}/metrics/{metric_name}"] = metric_value

    wandb.log(log_data)


def log_metrics_locally(
    all_metrics: Dict[str, Dict[str, float]], path_to_save: Path, epoch: int
) -> None:
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")
    metrics_df = metrics_df.round(4)
    metrics_df = metrics_df[["accuracy", "f1", "precision", "recall"]]

    tabulated_data = tabulate(metrics_df, headers="keys", tablefmt="pretty", showindex=True)
    if epoch:
        logger.info(f"Metrics on epoch {epoch}:\n{tabulated_data}\n")
    else:
        logger.info(f"Best epoch metrics:\n{tabulated_data}\n")

    if path_to_save:
        metrics_df.to_csv(path_to_save / "metrics.csv")


def save_metrics(train_metrics, metrics, loss, epoch, path_to_save, use_wandb) -> None:
    log_metrics_locally(
        all_metrics={"train": train_metrics, "val": metrics}, path_to_save=path_to_save, epoch=epoch
    )
    if use_wandb:
        wandb_logger(loss, train_metrics, epoch, mode="train")
        wandb_logger(None, metrics, epoch, mode="val")


def calculate_remaining_time(
    one_epoch_time, epoch_start_time, epoch, epochs, cur_iter, all_iters
) -> str:
    if one_epoch_time is None:
        average_iter_time = (time.time() - epoch_start_time) / cur_iter
        remaining_iters = epochs * all_iters - cur_iter

        hours, remainder = divmod(average_iter_time * remaining_iters, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}"

    time_for_remaining_epochs = one_epoch_time * (epochs + 1 - epoch)
    current_epoch_progress = time.time() - epoch_start_time
    hours, remainder = divmod(time_for_remaining_epochs - current_epoch_progress, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}"


def get_vram_usage():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )
        used, total = map(float, output.strip().split(", "))
        return round((used / total) * 100)
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
        return 0


def vis_one_image(image: np.ndarray, label: int, mode, score=None) -> None:
    if mode == "gt":
        prefix = "GT: "
        color = (46, 153, 60)
        postfix = ""
        position = (10, 30)
    elif mode == "pred":
        prefix = ""
        color = (148, 70, 44)
        postfix = f" {score:.2f}"
        position = (10, 50)

    cv2.putText(
        image,
        f"{prefix}{label_to_name_mapping[int(label)]}{postfix}",
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
        cv2.LINE_AA,
    )


def visualize(img_paths, batch_gt, batch_probs, dataset_path, path_to_save):
    """
    Saves images with class names.
      - Green text for GT
      - Blue text for preds
    """
    path_to_save.mkdir(parents=True, exist_ok=True)

    for gt, prob, img_path in zip(batch_gt, batch_probs, img_paths):
        img = cv2.imread(str(dataset_path / img_path))

        pred = torch.argmax(prob).item()
        label = gt.item()
        score = prob.max().item()

        vis_one_image(img, label, mode="gt")
        vis_one_image(img, pred, mode="pred", score=score)

        # Construct a filename and save
        outpath = path_to_save / Path(img_path).name
        cv2.imwrite(str(outpath), np.ascontiguousarray(img))
