# DORSA Segmentation

This folder integrates DORSA into a UNetFormer-style semantic segmentation pipeline.

## Dependency

```shell
conda create -n dorsa-seg python=3.8 -y
conda activate dorsa-seg
pip install -r requirements.txt
```

## Training

LoveDA example:

```shell
python train_supervision.py -c config/loveda/unetformer_dorsa_e36.py
```

## Inference

```shell
python loveda_test.py \
  -c config/loveda/unetformer_dorsa_e36.py \
  -o fig_results/loveda/unetformer_dorsa_e36 \
  -t d4
```

## Speed Test

```shell
python speed_test.py
```

Dataset paths and checkpoint paths in config files should be updated for your local environment.
