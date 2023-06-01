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

optimizer = dict(type='LARS', lr=2.0 * batch_ratio * update_interval, weight_decay=1e-6, momentum=0.9,
            param_group=[
                dict(name='(bn|gn|norm)(\d+)?.(weight|bias)', weight_decay=0., lars_exclude=True),
                dict(name='bias', weight_decay=0., lars_exclude=True)
            ])

lr_config = dict(type='iter', policy='cosinewarmup', T_max=total_epochs, warmup_iters=370, warmup_ratio=0.0001)

pretrained = None

Model = dict(
    type='SSModel',
    base_momentum=0.996,
    end_momentum=1.0
)

Evaluate="KNN"

embed_dim=512
output_dim=512
backbone=dict(
        type='ResNet',
        depth=18,
        zero_init_residual=True,
        initial=dict(distribution='uniform', mode="fan_out", nonlinearity="relu"),
)

head=dict(
    type="ICCHead",
    in_channels=embed_dim, 
    hidden_channels=512, 
    out_channels=output_dim, 
    proj_layers=3,
    warmup_epoch=[500, 600], 
    tau1=0.1, 
    tau2=0.07, 
    ratio=1.0,
    adaptive=True,
    pca=dict(dim=512, l2norm=True),
    initial=dict(nonlinearity="reset"),
)

knn=dict(    
    l2norm=True,
    topk=5
)

logger = dict(interval=15)
saver = dict(interval=500, saveType="epoch", maxsize=4)
evaluate_interval = 50
