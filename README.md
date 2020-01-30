# MOCO-tf2
This project implements constrastive learning algorithm introduced in [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) with tensorflow 2.0

## download dataset
The project trains ResNet50 with constrastive learning on imagenet resized version. Download the dataset with

```bash
python3 download_dataset.py
```

## training
Train the model with the following command

```bash
python3 train.py
```
