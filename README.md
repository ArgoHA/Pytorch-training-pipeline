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


## Integration info example
Model format: .tflite
Input size: Height=384, Width=384
Resize method: INTER_AREA (OpenCV) or equivalent
Input tensor shape: [Batch Size, Height, Width, Channels]
Channels order: RGB
Data type: float32

Preprocessing:
Read the image (BGR if using OpenCV), then convert to RGB
Resize to (384, 384) with interpolation method cv2.INTER_AREA
Convert to float32 and scale to 0–1
Normalize per channel
Add batch dimension
python snippet:

```
image = cv2.imread("img.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
image = image.astype(np.float32) / 255.0
image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
image = np.expand_dims(image, axis=0)
```
