import shutil
from pathlib import Path

import hydra
import onnx
import tensorflow as tf
import torch
from loguru import logger
from omegaconf import DictConfig
from onnx_tf.backend import prepare
from torch import nn

from src.dl.train import build_model
from src.dl.utils import get_latest_experiment_name

INPUT_NAME = "input"
OUTPUT_NAME = "output"


class ModelWithProcess(nn.Module):
    def __init__(self, original_model: nn.Module) -> None:
        super(ModelWithProcess, self).__init__()
        self.original_model = original_model

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        # Transpose from (BS, H, W, C) to (BS, C, H, W)
        input_image = input_image.permute(0, 3, 1, 2)
        output = self.original_model(input_image)
        return output


def load_model(model_name: str, model_path: Path, num_classes: int, device: str) -> nn.Module:
    model = build_model(
        model_name=model_name,
        pretrained=False,
        num_labels=num_classes,
        device=device,
        layers_to_train=-1,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def export_to_onnx(
    model: nn.Module, onnx_path: Path, x_test: torch.Tensor, max_batch_size: int, half: bool
) -> None:
    if half:
        model = model.half()
        x_test = x_test.half()
    if max_batch_size > 1:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    else:
        dynamic_axes = None
    torch.onnx.export(
        model,
        x_test,
        onnx_path,
        opset_version=16,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    logger.info("ONNX model exported")


def export_to_tf(onnx_model_path: Path, tf_path: str) -> None:
    tf_path = Path(tf_path)
    if tf_path.exists():
        shutil.rmtree(tf_path)
    onnx_model = onnx.load(str(onnx_model_path))
    tf_rep = prepare(onnx_model, auto_cast=True)
    tf_rep.export_graph(str(tf_path))


def export_to_tflite(tf_path: Path, tflite_path: str, half: bool) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if half:
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)
    model_path = Path(cfg.train.path_to_save) / "model.pt"
    num_classes = len(cfg.train.label_to_name)
    device = "cpu"

    model = load_model(cfg.model_name, model_path, num_classes, device)
    model = ModelWithProcess(model)
    model.eval()

    x_test = torch.randn(1, *cfg.train.img_size, 3, requires_grad=True).to(device)

    onnx_path = model_path.parent / "model.onnx"
    tf_path = model_path.parent / "tf"
    tflite_path = model_path.parent / "model.tflite"

    export_to_onnx(model, onnx_path, x_test, cfg.export.max_batch_size, cfg.export.half)
    export_to_tf(onnx_path, str(tf_path))
    export_to_tflite(tf_path, tflite_path, cfg.export.half)

    logger.info(f"Exports saved to: {model_path.parent}")


if __name__ == "__main__":
    main()
