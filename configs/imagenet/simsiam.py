num_class = 1000

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = [
    dict(type="MultiViewCrop", size_crops=[224], nmb_crops=[2], min_scale_crops=[0.2], max_scale_crops=[1.]),
    dict(type="ColorJitter", rand_apply=0.8, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    dict(type="RandomGrayscale", p=0.2),
    dict(type="GaussianBlur", rand_apply=0.5, sigma_min=0.1, sigma_max=2.0, kernel_size=23),
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
optimizer = dict(type='SGD', lr=0.05 * batch_ratio * update_interval, momentum=0.9, weight_decay=0.0001)
lr_config = dict(type='epoch', policy='cosine', T_max=total_epochs)

pretrained = None

Model = dict(
    type='SSModel'
)
Evaluate="KNN"
mixed_precision=True

embed_dim = 2048
backbone=dict(
        type='ResNet',
        depth=50,
        initial=dict(distribution="uniform", nonlinearity="relu", zerobias=True, mode="fan_out"),
        zero_init_residual=True)

head=dict(
    type="SimSiamHead",
    in_channels=embed_dim, 
    hidden_channels=512, 
    out_channels=embed_dim, 
    proj_layers=3,
)

knn=dict(    
    l2norm=True,
    topk=5
)

logger = dict(interval=100)
saver = dict(interval=10, saveType="epoch", maxsize=4)
evaluate_interval = 1