# DORSA Detection

This folder provides MMRotate-based oriented object detection configs for DORSA.

## Dependency

MMRotate depends on PyTorch, MMCV, and MMDetection. A typical setup is:

```shell
conda create -n dorsa-det python=3.8 -y
conda activate dorsa-det
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -U openmim
mim install mmcv-full
mim install mmdet
pip install -v -e .
```

Please also refer to the official MMRotate install guide if your CUDA/PyTorch version differs.

## Configs

| Task | Config | Detector | Angle | Schedule |
| --- | --- | --- | --- | --- |
| DOTA-v1.0 | `configs/dorsa/ORCNN_DORSA_T_fpn_le90_dota10_ss_e36.py` | Oriented R-CNN | le90 | 36 epochs |

## Training

```shell
bash tools/dist_train.sh configs/dorsa/ORCNN_DORSA_T_fpn_le90_dota10_ss_e36.py 2
```

To initialize from an ImageNet-1K pretrained DORSA backbone, set the `pretrained` field in the config to your local checkpoint path.

## Test-Set Formatting

```shell
bash tools/dist_test.sh \
  configs/dorsa/ORCNN_DORSA_T_fpn_le90_dota10_ss_e36.py \
  /path/to/checkpoint.pth \
  2 \
  --format-only \
  --eval-options submission_dir=/path/to/submission_dir
```

The resulting `Task1_*.txt` files can be zipped and uploaded to the DOTA evaluation server.

## Upstream

This detection code is based on MMRotate. Please follow the original MMRotate license and citation requirements when using this component.
