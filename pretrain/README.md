# DORSA ImageNet Pretrain

This folder is independent from `classification/`, `detection/`, `segmentation/`, and `change_detection/`.

## Dataset layout

`--data-path` should contain:

```text
imagenet/
  train/
    n01440764/*.JPEG
    ...
  val/
    n01440764/*.JPEG
    ...
```

## Single-GPU

```bash
python pretrain/main.py \
  --data-path /path/to/imagenet \
  --model DORSA_T_2262_s48 \
  --batch-size 128 \
  --epochs 300 \
  --lr 1e-3 \
  --amp \
  --output-dir ./outputs/imagenet_dorsa
```

## Multi-GPU (recommended)

```bash
torchrun --nproc_per_node=8 pretrain/main.py \
  --data-path /path/to/imagenet \
  --model DORSA_T_2262_s48 \
  --batch-size 256 \
  --epochs 300 \
  --lr 1e-3 \
  --amp \
  --sync-bn \
  --output-dir ./outputs/imagenet_dorsa
```

## Output files

- `checkpoint_last.pth`: latest checkpoint
- `checkpoint_best.pth`: best validation top-1 checkpoint
- `dorsa_t_imagenet_backbone.pth`: backbone-only weights (classifier head removed)
