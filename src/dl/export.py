from pathlib import Path

import hydra
import numpy as np
import onnx
import tensorflow as tf
import torch
import torch.nn.functional as F
import torch.onnx
from omegaconf import DictConfig
from onnx_tf.backend import prepare
from torch import nn

from src.dl.train import build_model
from src.utils import get_class_names


class ModelWithPreprocess(nn.Module):
    def __init__(self, original_model: nn.Module):
        super(ModelWithPreprocess, self).__init__()
        self.original_model = original_model

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        # Transpose from (BS, H, W, C) to (BS, C, H, W)
        input_image = input_image.permute(0, 3, 1, 2)

        output = self.original_model(input_image)
        output = F.softmax(output, dim=-1)
        output = output[0][1]
        return output


def load_model(
    model_path: Path, num_classes: int, device: str, model_name: str
) -> None:
    model = build_model(
        n_outputs=num_classes, device=device, model_name=model_name, layers_to_train=-1
    )
    model.eval()

    model.load_state_dict(torch.load(model_path))
    return model


def export_to_onnx(model: nn.Module, onnx_path: Path) -> None:
    x = torch.randn(1, 224, 224, 3, requires_grad=True)
    torch.onnx.export(
        model,
        x,
        onnx_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
    )


def export_to_tf(onnx_path: Path, tf_path: Path) -> None:
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_path)


def export_to_tflite(
    tf_path: Path,
    tflite_path: Path,
) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)

    # quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    model_path = Path(cfg.inference.model_path) / "model.pt"

    class_names = get_class_names(Path(cfg.train.data_path))
    num_classes = len(class_names)

    onnx_path = str(model_path.parent / "model.onnx")
    tf_path = str(model_path.parent / "model.pb")
    tflite_path = str(model_path.parent / "model.tflite")

    model = load_model(
        model_path, num_classes, cfg.inference.export_device, cfg.train.model_name
    )
    model = ModelWithPreprocess(model)

    export_to_onnx(model, onnx_path)
    export_to_tf(onnx_path, tf_path)
    export_to_tflite(tf_path, tflite_path)


if __name__ == "__main__":
    main()
