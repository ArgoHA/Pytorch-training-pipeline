# Pytorch Training Pipeline

This is an example of a PyTorch training pipeline that includes configurations with Hydra, tracking with WandB, and export options to ONNX, TensorRT, OpenVino, TensorFlow, and TensorFlow Lite.

## Configuration
- **project_name**: Name used in WandB.
- **exp**: Experiment name used across the repository for paths in training, exporting, and inference.
- **root**: Name of your local repository.
- **label_to_name**: Class names corresponding to your dataset's folder names.
- **use_scheduler**: Enables CyclicLR with two cycles per full training session.
- **layers_to_train**: Number of layers from the last one that require gradient computation (`-1` means no frozen layers).
- **cudnn_fixed**: Ensures reproducibility but can affect performance.
- **export.path_to_data**: Path to test data used in `infer.py`.

For model architecture, refer to [Torchvision](https://pytorch.org/vision/0.9/models.html) or [Hugging Face's timm](https://huggingface.co/timm).

## Usage
To run the scripts, use the following commands:
```bash
python -m src.etl.preprocess   # Converts images and PDFs to JPG format
python -m src.etl.split        # Creates train, validation, and test CSVs with image paths
python -m src.dl.train         # Runs the training pipeline
python -m src.dl.export        # Exports weights in various formats after training
python -m src.dl.infer         # Runs inference on a test folder with subfolders as classes
python -m src.dl.cnn_visualize # Creates a heatmap visualization based on the last convolutional layer gradients (Grad-CAM)
```

## Outputs
- **Best model**: Saved during the training process at `output/models/exp_name_date`.
- **Debug images**: Preprocessed images (including augmentations) are saved at `output/debug_img_processing` as they are fed into the model (except for normalization).
- **Visualizations**: `cnn_visualize` script saves heatmap visualizations of the last convolutional layer gradients at `output/visualized`.
