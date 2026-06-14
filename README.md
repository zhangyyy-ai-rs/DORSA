# Learning Where to Route: Dynamic Operator Routing with Structural Awareness for Efficient Remote Sensing Vision

Official PyTorch implementation of **DORSA**, a remote-sensing-oriented visual backbone for efficient recognition across multiple downstream tasks.

DORSA is designed for remote sensing imagery, where large homogeneous regions, object-scale variation, and dense spatial layouts make generic lightweight backbones less effective. This repository provides code for ImageNet pretraining and four representative remote-sensing tasks:

- Image classification
- Oriented object detection
- Semantic segmentation
- Change detection

## Repository Structure

```text
dorsa-main/
├── pretrain/           # ImageNet-1K pretraining entry
├── classification/     # Remote-sensing scene classification
├── detection/          # MMRotate-based oriented object detection
├── segmentation/       # GeoSeg/UNetFormer-based semantic segmentation
└── change_detection/   # A2Net/CLAFA-based change detection adapters
```

## Pretrained Weights

Before training downstream tasks, pre-trained DORSA weights from ImageNet-1K are placed into the task-specific weight path. Here we provide the DORSA pre-trained weights on the ImageNet-1K dataset for 300 epochs: [Download(https://drive.google.com/file/d/1G4O2Pio5N9Go8MRcrNgVt22N3_HvFy1t/view?usp=drive_link)

## ImageNet Pretraining

```shell
cd pretrain
bash train_imagenet.sh /path/to/imagenet ./outputs/imagenet_dorsa
```

The pretraining script exports:

- `checkpoint_last.pth`: full training checkpoint
- `dorsa_t_imagenet_backbone.pth`: backbone-only checkpoint for downstream tasks

## Oriented Object Detection

Detection is built on MMRotate. Install the detection package first:

```shell
cd detection
pip install -v -e .
```

Train DORSA with Oriented R-CNN on DOTA-v1.0:

```shell
bash tools/dist_train.sh configs/dorsa/ORCNN_DORSA_T_fpn_le90_dota10_ss_e36.py 2
```

For test-set formatting:

```shell
bash tools/dist_test.sh \
  configs/dorsa/ORCNN_DORSA_T_fpn_le90_dota10_ss_e36.py \
  /path/to/checkpoint.pth \
  2 \
  --format-only \
  --eval-options submission_dir=/path/to/submission_dir
```

## Image Classification

```shell
cd classification
python train.py --model DORSA_T_2262_s48 --datasets UCM-82
```

Supported dataset names in the provided script:

- `AID-82`
- `UCM-82`
- `RESISC45-82`

## Semantic Segmentation

```shell
cd segmentation
python train_supervision.py -c config/loveda/unetformer_dorsa_e36.py
```

## Change Detection

A2Net-style adapter:

```shell
cd change_detection/A2Net_DORSA
python train.py --backbone dorsa_t --file_root LEVIR
```

CLAFA-style adapter:

```shell
cd change_detection/CLAFA_DORSA
python main.py --backbone dorsa_t --dataset LEVIR_256_split
```

## Notes

- Dataset paths in the original training scripts are local defaults. Please update them to your environment before training.
- Checkpoints, logs, result folders, DOTA submission zips, and other generated artifacts are intentionally ignored by Git.
- The internal configuration name `DORSA_T_2262_s48` denotes the current tiny-size DORSA variant used in this codebase.

## Acknowledgements

This codebase builds on several excellent open-source projects, including [timm](https://github.com/huggingface/pytorch-image-models), [MMRotate](https://github.com/open-mmlab/mmrotate), [GeoSeg/UNetFormer](https://github.com/WangLibo1995/GeoSeg), [A2Net](https://github.com/guanyuezhen/A2Net), [CLAFA](https://github.com/xingronaldo/CLAFA), and the original LWGANet repository structure. We keep this acknowledgement to preserve proper upstream attribution while providing a cleaned DORSA release.

## Citation

The citation entry will be updated after the paper is public.
