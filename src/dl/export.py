from pathlib import Path

import hydra
import onnx
import openvino as ov
import tensorflow as tf
import tensorrt as trt
import torch
import torch.onnx
from omegaconf import DictConfig
from onnx_tf.backend import prepare
from torch import nn

from src.dl.train import build_model
from src.ptypes import class_names

INPUT_NAME = "input"
OUTPUT_NAME = "output"


def load_model(model_path: Path, num_classes: int, device: str) -> nn.Module:
    model = build_model(n_outputs=num_classes, device=device, layers_to_train=-1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def export_to_onnx(
    model: nn.Module, onnx_path: Path, x_test: torch.Tensor, max_batch_size: int
) -> None:
    if max_batch_size > 1:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    else:
        dynamic_axes = None
    torch.onnx.export(
        model,
        x_test,
        onnx_path,
        opset_version=12,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes=dynamic_axes,
    )


def export_to_openvino(model: nn.Module, ov_path: Path) -> None:
    core = ov.Core()
    ov_model = ov.convert_model(model)
    ov.save_model(ov_model, ov_path)


def export_to_tensorrt(
    onnx_file_path: Path, trt_path: Path, half: bool, max_batch_size: int
) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    if half:
        config.set_flag(trt.BuilderFlag.FP16)

    # Create an optimization profile for batching
    if max_batch_size > 1:
        profile = builder.create_optimization_profile()
        input_shape = network.get_input(0).shape
        min_shape = (1, *input_shape[1:])
        opt_shape = (max_batch_size, *input_shape[1:])
        max_shape = (max_batch_size, *input_shape[1:])
        profile.set_shape(INPUT_NAME, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise ValueError("Failed to build TensorRT engine")

    with open(trt_path, "wb") as f:
        f.write(engine)


def export_to_tf(onnx_path: Path, tf_path: str) -> None:
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_path)


def export_to_tflite(
    tf_path: Path,
    tflite_path: str,
) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    model_path = Path(cfg.export.model_path) / "model.pt"
    num_classes = len(class_names)

    trt_path = model_path.parent / "model.engine"
    onnx_path = model_path.parent / "model.onnx"
    ov_path = model_path.parent / "model.xml"
    tf_path = str(model_path.parent / "model.pb")
    tflite_path = str(model_path.parent / "model.tflite")

    model = load_model(model_path, num_classes, cfg.train.device)

    x_test = torch.randn(1, 3, *cfg.train.img_size, requires_grad=True).to(cfg.train.device)

    export_to_openvino(model, ov_path)
    export_to_onnx(model, onnx_path, x_test, cfg.export.max_batch_size)
    export_to_tensorrt(onnx_path, trt_path, cfg.export.trt_half, cfg.export.max_batch_size)
    export_to_tf(onnx_path, tf_path)
    export_to_tflite(tf_path, tflite_path)


if __name__ == "__main__":
    main()
