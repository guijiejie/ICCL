num_class = 1000

momentum = True

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = [
    dict(type="MultiViewCrop", size_crops=[224], nmb_crops=[2], min_scale_crops=[0.2], max_scale_crops=[1.]),
    dict(type="RandomHorizontalFlip"),
    dict(type="ColorJitter", rand_apply=0.8, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    dict(type="RandomGrayscale", p=0.2),
    dict(type="GaussianBlur", rand_apply=0.5, sigma_min=0.1, sigma_max=2.0, kernel_size=23),
    dict(type="ToTensor"),
    dict(type="Normalize", **img_norm_cfg)
]
transform_test = [
    dict(type="Resize", size=256),
    dict(type="CenterCrop", size=224),
    dict(type="ToTensor"),
    dict(type="Normalize", **img_norm_cfg)
]
batch_ratio = 4
dataset = dict(
    type='ImageNet',
    train_root='/apdcephfs/private_yuvalliu/imagenet/train',
    train_list='imagenet_label/train.txt',
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

optimizer = dict(type='LARS', lr=0.3 * batch_ratio * update_interval, weight_decay=1e-6, momentum=0.9,
            param_group=[
                dict(name='(bn|gn|norm)(\d+)?.(weight|bias)', weight_decay=0., lars_exclude=True),
                dict(name='bias', weight_decay=0., lars_exclude=True)
            ])
lr_config = dict(type='iter', policy='cosinewarmup', T_max=total_epochs, warmup_iters=12000, warmup_ratio=0.0001)

pretrained = None

Model = dict(
    type='{}SSModel'.format("Momentum" if momentum else ""),
    base_momentum=0.99,
    end_momentum=1.0
)
Evaluate="KNN"

embed_dim=2048
backbone=dict(
        type='ResNet',
        depth=50,
        initial=dict(distribution="uniform", nonlinearity="relu", zerobias=True, mode="fan_out"),
        zero_init_residual=True)

moco_dim = 256

neck=dict(
    type='NonlinearNeck',
    avgpool=False,
    layer_info=[
        dict(in_features=embed_dim, out_features=4096, norm=True, relu=True),
        dict(in_features=4096, out_features=4096, norm=True, relu=True),
        dict(in_features=4096, out_features=moco_dim, norm=True, relu=False),
    ],
    initial=dict(distribution='uniform', mode="fan_out", nonlinearity="reset"),
)

head=dict(
   type="MoCoHead",
    in_channels=moco_dim, 
    hidden_channels=4096, 
    out_channels=moco_dim,
    moco_buffer=None,
    tau=1.0,
    initial=dict(distribution='uniform', mode="fan_out", nonlinearity="reset"),
)

knn=dict(    
    l2norm=True,
    topk=5
)

logger = dict(interval=50)
saver = dict(interval=1, saveType="epoch", maxsize=4)
evaluate_interval = 1
