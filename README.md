# pytorch_training_pipeline
This is PyTorch training pipeline example with hydra configs, wandb and exports to ONNX, TF, TFLite

### Config
*project_name* - name used in wandb
*exp* - experiment name. It's used in wandb and more importantly everywhere in the repo (train/export/inference...) as a path for the experiment
*root* - your local repo name
*model_name* - in this repo I usesd two models - ShuffleNet_V2_X0_5, EfficientNet_B0 (shuffle_net, eficient_net_b0)

*layers_to_train* - how many layers from the last one to require gradient. `-1` = 0 frozen layers
