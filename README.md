# Pytorch training pipeline
This is PyTorch training pipeline example with hydra configs, wandb and exports to ONNX, TensorRT, OpenVino, TF, TFLite

### Config
*project_name* - name used in wandb

*exp* - experiment name. It's used in wandb and more importantly everywhere in the repo (train/export/inference...) as a path for the experiment

*root* - your local repo name

*layers_to_train* - how many layers from the last one to require gradient. `-1` = 0 frozen layers

*export.path_to_data* - test data used in infer.py

For model architecture use [Torchvision](https://pytorch.org/vision/0.9/models.html) or [Hugging Face's timm](https://huggingface.co/timm)

### Run
```
python -m src.etl.preprocess
python -m src.etl.split
python -m src.dl.train
python -m src.dl.export

python -m src.dl.infer
python -m src.dl.cnn_visualize
```

preprocess - convert images and pdfs to jpg

split - create train/val/test csvs with image paths

train - training pipeline

infer - run inferenc on test folder with subfolders = classes

export - after training is done, you can get weights in other formats

cnn_visualize - creates a heatmap visualisation based on last conv layer gradients
