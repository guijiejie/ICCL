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
batch_ratio=4
dataset = dict(
    type='ImageNet',
    train_root='/apdcephfs/private_yuvalliu/imagenet/train',
    train_list='imagenet_label/train_labeled.txt',
    test_root='/apdcephfs/private_yuvalliu/imagenet/val',
    test_list='imagenet_label/val_labeled.txt',
    batchsize=256 * batch_ratio,
    num_workers=48,
    num_class=num_class,
    trainmode="linear",
    transform_train=transform_train,
    testmode="linear",
    transform_test=transform_test,
    copy_dockerdata=True)

update_interval = 1

total_epochs = 100
optimizer = dict(type='SGD', lr=0.1 * batch_ratio, momentum=0.9, weight_decay=0)
lr_config = dict(policy='cosine', T_max=total_epochs)
# optimizer = dict(type='LARC', lr=0.1*batch_ratio, momentum=0.9, weight_decay=0)
# lr_config = dict(policy='cosine', T_max=total_epochs)

pretrained = None

Model=dict(
    type="Model"
)
Evaluate="CLS"

embed_dim = 384
backbone=dict(
        type='ViT',
        random_patch=True,
        patch_size=16,
        dim=embed_dim,
        heads=12,
        frozen=True
)

head=dict(
    type='LinearClsHead',
    num_classes=num_class,
    in_channels=embed_dim,
    topk=(1, 5),
)

evaluate_interval = 10
logger = dict(interval=100)
saver = dict(interval=50, saveType="epoch")