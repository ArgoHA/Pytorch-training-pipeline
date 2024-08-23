from collections import defaultdict
from pathlib import Path
from shutil import copyfile
from typing import Dict, List

import hydra
from omegaconf import DictConfig
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from src.dl.prod_infer import CustomModel
from src.ptypes import name_to_label_mapping


def run_prod_infer(model_path: Path, path_to_data: Path) -> List[int]:
    preds = []
    img_paths = [x for x in Path(path_to_data).glob("*.jpg")]

    model = CustomModel(model_path / "model.pt")

    output_path = Path("output") / "results" / path_to_data.name
    output_path.mkdir(exist_ok=True, parents=True)

    for img_path in tqdm(img_paths):
        img = Image.open(img_path)
        # img.draft("RGB", img_size)
        res = model.process(img)
        preds.append(name_to_label_mapping[res[0]])
        # copy_needed_class_for_new_dataset(img_path, res[0])

    print(f"{path_to_data}: {preds.count(0), preds.count(1), preds.count(2)}/{len(preds)}")
    return preds


def copy_needed_class_for_new_dataset(img_path, class_name):
    output_path = Path("output") / "pseudo_labeled" / class_name
    output_path.mkdir(exist_ok=True, parents=True)
    copyfile(img_path, output_path / img_path.name)


def compute_metrics(
    folder_predictions: defaultdict[int, List[int]], gt_val: Dict[str, int]
) -> None:
    all_preds = []
    all_true = []

    for folder_name, preds in folder_predictions.items():
        all_preds.extend(preds)
        all_true.extend([gt_val[folder_name]] * len(preds))

    precision = precision_score(all_true, all_preds, average="macro")
    recall = recall_score(all_true, all_preds, average="macro")
    f1 = f1_score(all_true, all_preds, average="macro")

    print(f"F1-Score: {round(f1, 3)}")
    print(f"Precision: {round(precision, 3)}")
    print(f"Recall: {round(recall, 3)}")


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    folder_predictions = defaultdict(list)
    for folder in Path(cfg.export.path_to_data).iterdir():
        if folder.is_dir():
            folder_predictions[folder.name] = run_prod_infer(
                model_path=Path(cfg.export.model_path),
                path_to_data=folder,
            )

    compute_metrics(folder_predictions, name_to_label_mapping)


if __name__ == "__main__":
    main()
