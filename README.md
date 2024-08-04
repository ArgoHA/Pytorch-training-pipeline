# Pytorch training pipeline
This is PyTorch training pipeline example with hydra configs, wandb and exports to ONNX, TF, TFLite

### Config
*project_name* - name used in wandb

*exp* - experiment name. It's used in wandb and more importantly everywhere in the repo (train/export/inference...) as a path for the experiment

*root* - your local repo name

*layers_to_train* - how many layers from the last one to require gradient. `-1` = 0 frozen layers

### Run
```
python -m src.etl.preprocess
python -m src.etl.split
python -m src.dl.train
python -m src.dl.export
```
