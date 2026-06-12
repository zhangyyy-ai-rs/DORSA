# CLAFA-DORSA Change Detection

This folder integrates DORSA into a CLAFA-style change detection pipeline.

## Train

```shell
python main.py --backbone dorsa_t --dataset LEVIR_256_split
```

## Test

```shell
python test.py --backbone dorsa_t --dataset LEVIR_256_split --checkpoint_dir ./checkpoints/dorsa_t/LEVIR_e200
```

Supported backbone options in this cleaned release are `dorsa_t`, `mobilenetv2`, and `resnet18d`.
