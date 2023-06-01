num_class = 10

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = [
    dict(type="RandomResizedCrop", size=128),
    dict(type="RandomHorizontalFlip"),
    dict(type="ToTensor"),
    dict(type="Normalize", **img_norm_cfg)
]
transform_test = [
    dict(type="Resize", size=160),
    dict(type="CenterCrop", size=128),
    dict(type="ToTensor"),
    dict(type="Normalize", **img_norm_cfg)
]
batch_ratio=16
dataset = dict(
    type='ImageNette',
    root="/mnt/ceph/home/yuvalliu/dataset/imagenette/imagenette2-160",
    batchsize=256 * batch_ratio,
    num_workers=32,
    trainmode="linear", 
    transform_train=transform_train, 
    testmode="linear",
    transform_test=transform_test)

update_interval = 1

total_epochs = 100
# optimizer = dict(type='SGD', lr=0.001 * batch_ratio, momentum=0.9, weight_decay=0, nesterov=True)
# lr_config = dict(policy='cosine', T_max=total_epochs)
optimizer = dict(type='LARC', lr=0.1*batch_ratio, momentum=0.9, weight_decay=0)
lr_config = dict(policy='cosine', T_max=total_epochs)

pretrained = None

Model=dict(
    type="Model"
)
Evaluate="CLS"

embed_dim = 384
backbone=dict(
    type='ViT',
    image_size=128, 
    patch_size=16, 
    dim=384,
    frozen=True,
)

head=dict(
    type='LinearClsHead',
    num_classes=num_class,
    in_channels=embed_dim,
    topk=(1, 5),
)

evaluate_interval = 10
logger = dict(interval=1)
saver = dict(interval=50, saveType="epoch")
