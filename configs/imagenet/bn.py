num_class = 1000

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = [
    dict(type="RandomResizedCrop", size=224),
    dict(type="RandomHorizontalFlip"),
    dict(type="ToTensor"),
    dict(type="Normalize", **img_norm_cfg)
]
transform_test = [
    dict(type="Resize", size=256),
    dict(type="CenterCrop", size=224),
    dict(type="ToTensor"),
    dict(type="Normalize", **img_norm_cfg)
]
batch_ratio = 2
dataset = dict(
    type='ImageNet',
    train_root='/apdcephfs/private_yuvalliu/imagenet/train',
    train_list='imagenet_label/train_labeled.txt',
    test_root='/apdcephfs/private_yuvalliu/imagenet/val',
    test_list='imagenet_label/val_labeled.txt',
    batchsize=128*batch_ratio,
    num_workers=64,
    num_class=num_class,
    trainmode="linear",
    transform_train=transform_train,
    testmode="linear",
    transform_test=transform_test)

total_epochs = 100
optimizer = dict(type='SGD', lr=0.05 * batch_ratio, momentum=0.9, weight_decay=0.0001)
lr_config = dict(policy='step', milestones=[30, 60, 90])

SyncBN = False
pretrained = None
Model=dict(
    type="Model"
)
Evaluate="CLS"

backbone=dict(
        type='ResNet',
        depth=50,
        initial=dict(distribution="uniform", nonlinearity="relu", zerobias=True, mode="fan_out"),
        zero_init_residual=True)

head=dict(
    type='LinearClsHead',
    num_classes=num_class,
    in_channels=2048,
    topk=(1, 5),
)

logger = dict(interval=100)
saver = dict(interval=50)
