# Pytorch Training Pipeline

This is an example of a PyTorch training pipeline that includes configurations with Hydra, tracking with WandB, and export options to ONNX, TensorRT, OpenVino, TensorFlow, and TensorFlow Lite.

## Configuration
- **model_name** - Model name from timm
- **root** - Path to the directory where you store your dataset and where model outputs will be saved
- **data_path** - Path to the dataset filder. Should contain subfolders with label names
- **label_to_name**: Class names corresponding to your dataset's folder names.
- **project_name**: Name used in WandB.
- **exp**: Experiment name used across the repository for paths in training, exporting, and inference.

For model architecture, refer to [Torchvision](https://pytorch.org/vision/0.9/models.html) or [Hugging Face's timm](https://huggingface.co/timm).

## Usage
To run the scripts, use the following commands:
```bash
python -m src.etl.preprocess   # Converts images and PDFs to JPG format
python -m src.etl.split        # Creates train, validation, and test CSVs with image paths
python -m src.dl.train         # Runs the training pipeline
python -m src.dl.export        # Exports weights in various formats after training
python -m src.dl.bench         # Runs all exported models on the test set
python -m src.dl.infer         # Runs inference on a test folder with subfolders as classes
python -m src.dl.cnn_visualize # Creates a heatmap visualization based on the last convolutional layer gradients (Grad-CAM)
```

## Inference
Use inference classes in `src/infer`. Currently available:
- Torch
- TensorRT
- OpenVINO

## Outputs
- **Best model**: Saved during the training process at `output/models/exp_name_date`.
- **Debug images**: Preprocessed images (including augmentations) are saved at `output/debug_images` as they are fed into the model (except for normalization).
- **Evaluation predicts**: Visualised model's predictions on val set. Includes GT as green and preds as blue.
- **Visualizations**: `cnn_visualize` script saves heatmap visualizations of the last convolutional layer gradients at `output/visualized`.
