# Classifier

This is an pipeline for training/exporting/inferencing image classification models with PyTirch and TIMM. Hydra configs, WandB for experiment tracking, exporting to ONNX, TensorRT, OpenVino, TensorFlow, and TensorFlow Lite.

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


## Models comparison
Tested on 13k images dataset `train: 10440, val: 1306, test: 1304`, 4 classes, input size 348. Inferenced on 5070ti without optimisations, pure PyyTorch. F1 score calculated on the test set:

```
model_name | latency |  F1
-----------+---------+-------
efnet_b0   | 6.5ms   | 0.969
efnet_b1   | 7.6ms   | 0.976
efnetv2_s  | 9.4ms   | 0.967
mobnet4s   | 5.6ms   | 0.948
mobnet4m   | 6.7ms   | 0.970
mobnet4l   | 7.7ms   | 0.971
fastvit    | 7.2ms   | 0.973
```
