# A2Net-DORSA Change Detection

This folder integrates DORSA into an A2Net-style change detection pipeline.

## Train

```shell
python train.py --backbone dorsa_t --file_root LEVIR
```

Supported dataset keys in the script include `LEVIR`, `SYSU`, `WHUCD256`, `CDD`, and `quick_start`.

## Test

```shell
python test.py --backbone dorsa_t --file_root LEVIR --savedir ./results
```

Place DORSA pretrained checkpoints under `backbone_weights/` if `--pretrained True` is used.
