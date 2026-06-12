from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.loveda_dataset import *
from geoseg.models.UNetFormer_dorsa import UNetFormer_DORSA
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 36
ignore_index = len(CLASSES)
train_batch_size = 16
val_batch_size = 16

lr = 3.5e-4
weight_decay = 0.01

# backbone 的参数组名称仍然是 backbone.*，
# 因为 UNetFormer_DORSA 里主干参数名仍是 backbone.*
backbone_lr = 3.5e-4
backbone_weight_decay = 0.01

num_classes = len(CLASSES)
classes = CLASSES

weights_name = "unetformer-dorsa-512crop-ms-epoch30-rep"
weights_path = "model_weights/loveda/{}".format(weights_name)
test_weights_name = "last"
log_name = "loveda/{}".format(weights_name)

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1

# pretrained_ckpt_path 是 Lightning checkpoint 路径，不是 backbone 预训练路径
pretrained_ckpt_path = None

# default or gpu ids:[0] or gpu nums: 2
gpus = "auto"

# whether continue training with the checkpoint, default None
resume_ckpt_path = None


# define the network
net = UNetFormer_DORSA(
    num_classes=num_classes,
    decode_channels=64,
    dropout=0.1,
    window_size=8,

    # DORSA backbone setting
    # 如果你在 unet_lorf.py 里默认就是这些，也可以只写 num_classes=num_classes
    stem_dim=48,
    depths=(2, 2, 6, 2),
    mlp_ratio=2.0,
    hidden_ratios=(0.5, 0.5, 0.5, 0.5),
    num_atoms=(4, 4, 8, 8),
    topks=(1, 1, 2, 2),
    drop_path_rate=0.1,
    temperature=1.25,
    norm_layer=dict(type="SyncBN", requires_grad=True),
    pretrained=None,
    use_stage1_cell=True,
    use_stage2_cell=True,
    stage1_position="mid",
    stage2_position="pre"
)


# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True


# define the dataloader
def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([
        RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode="value"),
        SmartCropV1(
            crop_size=512,
            max_ratio=0.75,
            ignore_index=ignore_index,
            nopad=False
        )
    ])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug["image"], aug["mask"]
    return img, mask


train_dataset = LoveDATrainDataset(
    transform=train_aug,
    data_root="/path/to/loveda/Train"
)

val_dataset = loveda_val_dataset

test_dataset = LoveDATestDataset()


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)


# define the optimizer
# 这里会单独匹配 UNetFormer_DORSA.backbone 里的参数
layerwise_params = {
    "backbone.*": dict(
        lr=backbone_lr,
        weight_decay=backbone_weight_decay
    )
}

net_params = utils.process_model_params(
    net,
    layerwise_params=layerwise_params
)

base_optimizer = torch.optim.AdamW(
    net_params,
    lr=lr,
    weight_decay=weight_decay
)

optimizer = Lookahead(base_optimizer)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_epoch,
    eta_min=1e-6
)
