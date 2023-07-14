from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import loguru
import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.utils import get_class_names


def get_stats(root_path: Path) -> Tuple[List[Path], List[int], Dict[str, int]]:
    image_paths = []
    labels = []
    classes = {}

    im_folder_names = get_class_names(root_path)
    label = 0

    for class_name in im_folder_names:
        class_dir = root_path / class_name
        if str(class_dir.name).startswith("."):  # skip .folders
            continue

        for file_name in class_dir.iterdir():  # for every image in a class
            if str(file_name.name).startswith("."):  # skip .DS_store
                continue
            classes[class_dir.name] = classes.get(class_dir.name, 0) + 1

            image_paths.append(class_dir / file_name)
            labels.append(label)

        label += 1

    print("Count images per class:", classes)
    return image_paths, labels, classes


def split(
    data_pat: Path,
    train_split: float,
    val_split: float,
    image_paths: List[Path],
    labels: List[int],
) -> None:
    test_split = 1 - train_split - val_split
    if test_split <= 0.001:
        test_split = 0

    indices = np.arange(len(image_paths))
    train_idxs, temp_idxs = train_test_split(
        indices, test_size=(1 - train_split), stratify=labels
    )

    if test_split:
        test_idxs, val_idxs = train_test_split(
            temp_idxs,
            test_size=(val_split / (val_split + test_split)),
            stratify=[labels[i] for i in temp_idxs],
        )
    else:
        val_idxs = temp_idxs
        test_idxs = []

    splits = {"train": train_idxs, "val": val_idxs}
    if test_split:
        splits["test"] = test_idxs

    for split_name, split in splits.items():
        with open(data_pat / f"{split_name}.csv", "w") as f:
            for num, idx in enumerate(split):
                f.write(str(image_paths[idx]) + "," + str(labels[idx]) + "\n")
            loguru.logger.info(f"{split_name}: {num}")


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    data_path = Path(cfg.train.data_path)

    image_paths, labels, _ = get_stats(data_path)
    split(data_path, cfg.train.train_split, cfg.train.val_split, image_paths, labels)


if __name__ == "__main__":
    main()
