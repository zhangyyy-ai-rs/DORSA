import argparse
import datetime
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "classification"))
from models.dorsa import DORSA_T_2262_s48  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("DORSA ImageNet Pretrain")
    parser.add_argument("--data-path", type=str, required=True, help="ImageNet root folder, containing train/ and val/")
    parser.add_argument("--model", type=str, default="DORSA_T_2262_s48", choices=["DORSA_T_2262_s48"])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp")
    parser.add_argument("--sync-bn", action="store_true", help="Use SyncBN inside DORSA")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="./outputs/imagenet_dorsa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-freq", type=int, default=100)
    return parser.parse_args()


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    return distributed, rank, local_rank, world_size


def is_main_process(rank):
    return rank == 0


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        if self.count == 0:
            return 0.0
        return self.sum / self.count


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


@torch.no_grad()
def accuracy(output, target, topk=(1, 5)):
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        k = min(k, output.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def build_model(args, device):
    norm_layer = dict(type="SyncBN", requires_grad=True) if args.sync_bn else dict(type="BN", requires_grad=True)
    model = DORSA_T_2262_s48(norm_layer=norm_layer, fork_feat=False)
    if model.out_head.out_features != args.num_classes:
        in_features = model.out_head.in_features
        model.out_head = nn.Linear(in_features, args.num_classes)
    model.to(device)
    return model


def build_dataloader(args, distributed):
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(args.img_size * 256 / 224)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    val_set = datasets.ImageFolder(val_dir, transform=val_transform)

    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, train_sampler


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch, args, distributed, world_size, rank):
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    start = time.time()
    for i, (images, target) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(images)
            loss = criterion(output, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if distributed:
            loss = reduce_tensor(loss.detach(), world_size)
            acc1 = reduce_tensor(acc1.detach(), world_size)
            acc5 = reduce_tensor(acc5.detach(), world_size)

        bsz = images.size(0)
        loss_meter.update(loss.item(), bsz)
        top1_meter.update(acc1.item(), bsz)
        top5_meter.update(acc5.item(), bsz)

        if is_main_process(rank) and (i % args.print_freq == 0):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Train Epoch [{epoch}] [{i}/{len(loader)}] "
                f"loss={loss_meter.avg:.4f} top1={top1_meter.avg:.2f} top5={top5_meter.avg:.2f} lr={lr:.6f}"
            )

    if is_main_process(rank):
        dt = time.time() - start
        print(f"Train Epoch [{epoch}] done in {dt:.1f}s: loss={loss_meter.avg:.4f}, top1={top1_meter.avg:.2f}, top5={top5_meter.avg:.2f}")

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, args, distributed, world_size, rank):
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for i, (images, target) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(images)
            loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if distributed:
            loss = reduce_tensor(loss.detach(), world_size)
            acc1 = reduce_tensor(acc1.detach(), world_size)
            acc5 = reduce_tensor(acc5.detach(), world_size)

        bsz = images.size(0)
        loss_meter.update(loss.item(), bsz)
        top1_meter.update(acc1.item(), bsz)
        top5_meter.update(acc5.item(), bsz)

    if is_main_process(rank):
        print(f"Val: loss={loss_meter.avg:.4f}, top1={top1_meter.avg:.2f}, top5={top5_meter.avg:.2f}")

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_acc1, output_dir, filename, distributed):
    state_dict = model.module.state_dict() if distributed else model.state_dict()
    ckpt = {
        "epoch": epoch,
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_acc1": best_acc1,
    }
    torch.save(ckpt, os.path.join(output_dir, filename))

    # Optional backbone-only weights for downstream (without classifier head)
    backbone_only = {k: v for k, v in state_dict.items() if not k.startswith("out_head.")}
    torch.save({"model": backbone_only}, os.path.join(output_dir, "dorsa_t_imagenet_backbone.pth"))


def main():
    args = parse_args()
    distributed, rank, local_rank, world_size = init_distributed()
    set_seed(args.seed + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    if is_main_process(rank):
        print("Args:", args)
        print("Distributed:", distributed, "world_size:", world_size)
        print("Start time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    model = build_model(args, device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    train_loader, val_loader, train_sampler = build_dataloader(args, distributed)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp) if torch.cuda.is_available() else None

    start_epoch = 0
    best_acc1 = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        target = model.module if distributed else model
        target.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc1 = ckpt.get("best_acc1", 0.0)
        if is_main_process(rank):
            print(f"Resumed from {args.resume}, start_epoch={start_epoch}, best_acc1={best_acc1:.2f}")

    for epoch in range(start_epoch, args.epochs):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, args, distributed, world_size, rank)
        _, acc1, _ = evaluate(model, val_loader, criterion, device, args, distributed, world_size, rank)
        scheduler.step()

        if is_main_process(rank):
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_acc1, args.output_dir, "checkpoint_last.pth", distributed)
            if acc1 > best_acc1:
                best_acc1 = acc1
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_acc1, args.output_dir, "checkpoint_best.pth", distributed)
            print(f"Epoch {epoch} finished. best_acc1={best_acc1:.2f}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
