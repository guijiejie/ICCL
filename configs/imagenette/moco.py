img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train = [[
    dict(type="RandomResizedCrop", size=128),
    dict(type="ColorJitter", rand_apply=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    dict(type="RandomGrayscale", p=0.2),
    dict(type="GaussianBlur", rand_apply=1.0, sigma_min=0.1, sigma_max=2.0),
    dict(type="ToTensor"),
    dict(type="ColorPermutation", rand_apply=0.8),
    dict(type="RandomHorizontalFlip"),
    dict(type="Normalize", **img_norm_cfg)
],[
    dict(type="RandomResizedCrop", size=128),
    dict(type="ColorJitter", rand_apply=1.0, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    dict(type="RandomGrayscale", p=0.2),
    dict(type="GaussianBlur", rand_apply=0.1, sigma_min=0.1, sigma_max=2.0),
    dict(type="Solarization", rand_apply=0.2),
    dict(type="ToTensor"),
    dict(type="ColorPermutation", rand_apply=0.8),
    dict(type="RandomHorizontalFlip"),
    dict(type="Normalize", **img_norm_cfg)
]]

transform_test = [
    dict(type="Resize", size=160),
    dict(type="CenterCrop", size=128),
    dict(type="ToTensor"),
    dict(type="Normalize", **img_norm_cfg)
]

batch_ratio = 1

dataset = dict(
    type='ImageNette',
    root="/mnt/ceph/home/yuvalliu/dataset/imagenette/imagenette2-160",
    batchsize=256 * batch_ratio,
    num_workers=32,
    trainmode="ss", 
    transform_train=transform_train, 
    testmode="linear",
    transform_test=transform_test)

update_interval = 1

total_epochs = 1000

pretrained = None

Model = dict(
    type='MomentumSSModel',
    base_momentum=0.996,
    end_momentum=1.0
)

Evaluate="KNN"

ViTBackbone = True
pretrained = None

if ViTBackbone:
    optimizer = dict(type='Adam', lr=1e-3 * batch_ratio * update_interval, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)
    lr_config = dict(type='epoch', policy='cosinewarmup', T_max=total_epochs, warmup_iters=40, warmup_ratio=0.0001)

    embed_dim = 384
    backbone=dict(
        type='ViT',
        image_size=128, 
        patch_size=16, 
        dim=384,
        random_patch=True
    )

    moco_temperature = 0.2
else:
    optimizer = dict(type='LARS', lr=2.0 * batch_ratio * update_interval, weight_decay=1e-6, momentum=0.9,
                param_group=[
                    dict(name='(bn|gn|norm)(\d+)?.(weight|bias)', weight_decay=0., lars_exclude=True),
                    dict(name='bias', weight_decay=0., lars_exclude=True)
                ])
    lr_config = dict(type='iter', policy='cosinewarmup', T_max=total_epochs, warmup_iters=370, warmup_ratio=0.0001)

    embed_dim=512
    backbone=dict(
            type='ResNet',
            depth=18,
            zero_init_residual=True,
            initial=dict(distribution='uniform', mode="fan_out", nonlinearity="relu"),
    )

    moco_temperature = 1.0

moco_dim = 256
hidden_dim = 4096

neck=dict(
    type='NonlinearNeck',
    avgpool=False,
    layer_info=[
        dict(in_features=embed_dim, out_features=hidden_dim, norm=True, relu=True),
        dict(in_features=hidden_dim, out_features=hidden_dim, norm=True, relu=True),
        dict(in_features=hidden_dim, out_features=moco_dim, norm=True, relu=False),
    ],
    initial=dict(distribution='uniform', mode="fan_out", nonlinearity="reset"),
)

head=dict(
   type="MoCoHead",
    in_channels=moco_dim, 
    hidden_channels=hidden_dim, 
    out_channels=moco_dim,
    moco_buffer=dict(queue_size=4096),
    tau=moco_temperature,
    initial=dict(distribution='uniform', mode="fan_out", nonlinearity="reset"),
)

knn=dict(
    l2norm=True,
    topk=5
)

logger = dict(interval=15)
saver = dict(interval=500, saveType="epoch", maxsize=4)
evaluate_interval = 50
