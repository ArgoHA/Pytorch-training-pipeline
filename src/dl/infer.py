from collections import defaultdict
from pathlib import Path
from shutil import copyfile
from typing import Dict, List

import cv2
import hydra
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from src.infer.torch_model import Torch_model
from src.ptypes import img_size, label_to_name_mapping, name_to_label_mapping

THRESHOLD = 0.5


def run_prod_infer(model: Torch_model, path_to_data: Path) -> List[int]:
    preds = []
    img_paths = [x for x in Path(path_to_data).glob("*.jpg")]

    output_path = Path("output") / "results" / path_to_data.name
    output_path.mkdir(exist_ok=True, parents=True)

    for img_path in tqdm(img_paths):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = model(img)
        preds.append(res[0])
        # copy_needed_class_for_new_dataset(img_path, res[0])
    return preds


def copy_needed_class_for_new_dataset(img_path, class_name):
    output_path = Path("output") / "pseudo_labeled" / class_name
    output_path.mkdir(exist_ok=True, parents=True)
    copyfile(img_path, output_path / img_path.name)


def compute_metrics(
    folder_predictions: defaultdict[int, List[int]], gt_val: Dict[str, int]
) -> None:
    all_preds = []
    all_gt = []

    num_labels = len(label_to_name_mapping)
    if num_labels == 2:
        average = "binary"
    else:
        average = "macro"

    for folder_name, preds in folder_predictions.items():
        all_preds.extend(preds)
        all_gt.extend([gt_val[folder_name]] * len(preds))

    precision = precision_score(all_gt, all_preds, average=average)
    recall = recall_score(all_gt, all_preds, average=average)
    f1 = f1_score(all_gt, all_preds, average=average)
    accuracy = accuracy_score(all_gt, all_preds)

    print(f"F1-Score: {round(f1, 3)}")
    print(f"Precision: {round(precision, 3)}")
    print(f"Recall: {round(recall, 3)}")
    print(f"Accuracy: {round(accuracy, 3)}")


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    folder_predictions = defaultdict(list)
    model = Torch_model(
        model_name=cfg.model_name,
        model_path=Path(cfg.train.path_to_save) / "model.pt",
        n_outputs=len(label_to_name_mapping),
        input_size=img_size,
    )
    for folder in Path(cfg.train.path_to_test_data).iterdir():
        if folder.is_dir():
            folder_predictions[folder.name] = run_prod_infer(model=model, path_to_data=folder)

    compute_metrics(folder_predictions, name_to_label_mapping)


if __name__ == "__main__":
    main()
