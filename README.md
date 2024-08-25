# Pytorch training pipeline
This is PyTorch training pipeline example with hydra configs, wandb and exports to ONNX, TensorRT, OpenVino, TF, TFLite

### Config
*project_name* - name used in wandb

*exp* - experiment name. It's used in wandb and more importantly everywhere in the repo (train/export/inference...) as a path for the experiment

*root* - your local repo name

*label_to_name* - classes to be trained on. Should be the same as you dataset folders naming

*use_scheduler* - turns on CyclicLR with 2 cycles per full training

*layers_to_train* - how many layers from the last one to require gradient. `-1` = 0 frozen layers

*cudnn_fixed* - used for reproducibillity. Can affect performance.

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


### Created output
Best model is being saved during the training process to `output/models/exp_name_date`, same for exported models.

Flag `debug_img_processing` saves preprocessed (including augmentations) images to `output/debug_img_processing` as they are fed to the model (except for normalization).

Script `cnn_visualize` saves heatmap visualisation of last conv layer grads for `output/visualized`
