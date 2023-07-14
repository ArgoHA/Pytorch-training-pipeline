from pathlib import Path
from typing import List

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from src.dl.train import build_model
from src.utils import get_class_names


def get_transforms(img: Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(img)


def infer(model: nn.Module, img: torch.Tensor, device: str) -> torch.Tensor:
    inputs = get_transforms(img)[None].to(device)
    logits = model(inputs)
    probs = torch.softmax(logits, dim=1)
    return probs


def prepare_model(
    model_path: Path, num_classes: int, device: str, model_name: str
) -> nn.Module:
    model = build_model(num_classes, device, model_name=model_name, layers_to_train=-1)
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def run_infer(
    model_path: Path,
    num_classes: int,
    device: str,
    model_name,
    path_to_data: Path,
    class_names: List[str],
) -> None:
    out = []
    model = prepare_model(model_path, num_classes, device, model_name)
    img_paths = [x for x in Path(path_to_data).glob("*.jpg")]
    output_path = path_to_data / "predicts"
    output_path.mkdir(exist_ok=True)

    for img_path in tqdm(img_paths):
        img = Image.open(img_path)
        probs = infer(model, img, device)
        res = torch.argmax(probs).item()
        out.append(res)

        pred_class = class_names[res]

        img.save(output_path / f"{pred_class.upper()}_{img_path.name}")


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    class_names = get_class_names(Path(cfg.train.data_path))
    num_classes = len(class_names)

    for folder in Path(cfg.inference.path_to_data).iterdir():
        if folder.is_dir():
            run_infer(
                model_path=Path(cfg.inference.model_path) / "model.pt",
                num_classes=num_classes,
                device=cfg.train.device,
                model_name=cfg.train.model_name,
                path_to_data=folder,
                class_names=class_names,
            )


if __name__ == "__main__":
    main()
