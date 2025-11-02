from pathlib import Path

import hydra
import openvino as ov
import tensorrt as trt
import torch
import torch.onnx
from loguru import logger
from omegaconf import DictConfig
from torch import nn

from src.dl.train import build_model
from src.dl.utils import get_latest_experiment_name

INPUT_NAME = "input"
OUTPUT_NAME = "output"


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
        opset_version=19,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes=dynamic_axes,
        dynamo=True,
    ).save(onnx_path)

    logger.info("ONNX model exported")


def export_to_openvino(torch_model: nn.Module, ov_path: Path) -> None:
    model = ov.convert_model(input_model=torch_model)
    # rename inputs and outputs
    model.inputs[0].tensor.set_names({INPUT_NAME})
    model.outputs[0].tensor.set_names({OUTPUT_NAME})
    ov.serialize(model, str(ov_path.with_suffix(".xml")), str(ov_path.with_suffix(".bin")))
    logger.info("OpenVINO model exported")


def export_to_tensorrt(
    onnx_file_path: Path, trt_path: Path, half: bool, max_batch_size: int
) -> None:
    tr_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(tr_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, tr_logger)

    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
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
    logger.info("TensorRT model exported")


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)
    model_path = Path(cfg.train.path_to_save) / "model.pt"
    num_classes = len(cfg.train.label_to_name)

    trt_path = model_path.parent / "model.engine"
    onnx_path = model_path.parent / "model.onnx"
    ov_path = model_path.parent / "model.xml"

    model = load_model(cfg.model_name, model_path, num_classes, cfg.train.device)

    x_test = torch.randn(1, 3, *cfg.train.img_size, requires_grad=True).to(cfg.train.device)

    export_to_onnx(model, onnx_path, x_test, cfg.export.max_batch_size, cfg.export.half)
    export_to_openvino(model, ov_path)
    export_to_tensorrt(onnx_path, trt_path, cfg.export.half, cfg.export.max_batch_size)

    logger.info(f"Exports saved to: {model_path.parent}")


if __name__ == "__main__":
    main()
