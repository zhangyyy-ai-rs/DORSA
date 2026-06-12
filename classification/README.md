# DORSA Classification

This folder contains the remote-sensing scene classification entry for DORSA.

## Dependency

```shell
conda create -n dorsa-cls python=3.9 -y
conda activate dorsa-cls
pip install torch torchvision timm ptflops tensorboard tqdm matplotlib pillow
```

## Dataset

The default script supports:

- `AID-82`
- `UCM-82`
- `RESISC45-82`

Update dataset paths in `utils.py` before training.

## Training

```shell
python train.py --model DORSA_T_2262_s48 --datasets UCM-82
```

## Speed Test

```shell
python speed_test.py
```

The model name `DORSA_T_2262_s48` denotes the current tiny-size DORSA classification variant.
