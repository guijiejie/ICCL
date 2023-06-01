num_class = 1000

momentum = False

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
batch_ratio = 2
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
optimizer = dict(type='Adam', lr=1.5e-4 * batch_ratio * update_interval, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
lr_config = dict(type='epoch', policy='cosinewarmup', T_max=total_epochs, warmup_iters=20, warmup_ratio=0.0001)

pretrained = None

Model = dict(
    type='TwoSSModel',
    base_momentum=0.99,
    end_momentum=1.0,
    SwapTrans=True,
)
Evaluate="KNN"

backbone=dict(
        type='ResNet',
        depth=50,
        initial=dict(distribution="uniform", nonlinearity="relu", zerobias=True, mode="fan_out"),
        zero_init_residual=True)
neck=dict(
    type='NonlinearNeck',
    avgpool=False,
    layer_info=[
        dict(in_features=2048, out_features=4096, norm=True, relu=True),
        dict(in_features=4096, out_features=4096, norm=True, relu=True),
        dict(in_features=4096, out_features=2048, norm=True, relu=False),
    ],
    initial=dict(distribution='uniform', mode="fan_out", nonlinearity="relu"),
)

backbone2=dict(
        type='ViT',
        random_patch=True,
        patch_size=16,
        dim=384,
        heads=12
)
neck2=dict(
    type='NonlinearNeck',
    avgpool=False,
    layer_info=[
        dict(in_features=384, out_features=4096, norm=True, relu=True),
        dict(in_features=4096, out_features=4096, norm=True, relu=True),
        dict(in_features=4096, out_features=2048, norm=True, relu=False),
    ],
    initial=dict(distribution='uniform', mode="fan_out", nonlinearity="relu"),
)

head=dict(
   type="CoTrainHead",
    in_channels=2048, 
    hidden_channels=4096, 
    out_channels=2048,
    initial=dict(distribution='uniform', mode="fan_out", nonlinearity="reset"),
)

knn=dict(    
    l2norm=True,
    topk=5
)

logger = dict(interval=50)
saver = dict(interval=1, saveType="epoch", maxsize=4)
evaluate_interval = 1
