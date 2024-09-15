import time
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.dl.train import get_metrics
from src.infer.ov_model import OV_model
from src.infer.torch_model import Torch_model
from src.infer.trt_model import TensorRT_model
from src.ptypes import img_size, label_to_name_mapping, name_to_label_mapping


class CustomDataset(Dataset):
    def __init__(
        self,
        img_size: Tuple[int, int],
        root_path: Path,
        split: pd.DataFrame,
    ) -> None:
        self.root_path = root_path
        self.split = split
        self.img_size = img_size

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        image_path, label = self.split.iloc[idx]
        return image_path, label

    def __len__(self) -> int:
        return len(self.split)


def test_model(test_loader: DataLoader, data_path: Path, model, name: str):
    logger.info(f"Testing {name} model")
    predictions = []
    gt_labels = []
    latency = []

    for batch in tqdm(test_loader, total=len(test_loader)):
        image_paths, labels = batch
        batch_predictions = []
        for image_path in image_paths:
            image = Image.open(data_path / image_path)
            image.draft("RGB", img_size)
            image = ImageOps.exif_transpose(image)

            t0 = time.perf_counter()
            class_name, max_prob = model(image)
            latency.append(time.perf_counter() - t0)

            predicted_label = name_to_label_mapping[class_name]
            batch_predictions.append(predicted_label)

        predictions.extend(batch_predictions)
        gt_labels.extend(labels.tolist())

    metrics = get_metrics(gt_labels, predictions)
    metrics["latency"] = np.mean(latency[1:])
    return metrics


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    data_path = Path(cfg.train.data_path)

    torch_model = Torch_model(
        model_path=str(Path(cfg.export.model_path) / "model.pt"),
        label_to_name=label_to_name_mapping,
        input_width=img_size[1],
        input_height=img_size[0],
        half=cfg.export.half,
    )

    trt_model = TensorRT_model(
        model_path=str(Path(cfg.export.model_path) / "model.engine"),
        label_to_name=label_to_name_mapping,
        input_width=img_size[1],
        input_height=img_size[0],
        half=cfg.export.half,
    )
    ov_model = OV_model(
        model_path=str(Path(cfg.export.model_path) / "model.xml"),
        label_to_name=label_to_name_mapping,
        input_width=img_size[1],
        input_height=img_size[0],
        half=cfg.export.half,
    )

    test_dataset = CustomDataset(
        img_size=img_size,
        root_path=data_path,
        split=pd.read_csv(data_path / "test.csv", header=None),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    all_metrics = {}
    models = {
        "torch": torch_model,
        "TensorRT": trt_model,
        "OV": ov_model,
    }
    for model_name, model in models.items():
        all_metrics[model_name] = test_model(test_loader, data_path, model, model_name)

    metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")
    print(metrics_df)


if __name__ == "__main__":
    main()
