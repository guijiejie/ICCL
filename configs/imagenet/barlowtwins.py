num_class = 1000

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train = [[
    dict(type="RandomResizedCrop", size=224),
    dict(type="ColorJitter", rand_apply=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    dict(type="RandomGrayscale", p=0.2),
    dict(type="GaussianBlur", rand_apply=1.0, sigma_min=0.1, sigma_max=2.0),
    dict(type="ToTensor"),
    dict(type="RandomHorizontalFlip"),
    dict(type="Normalize", **img_norm_cfg)
],[
    dict(type="RandomResizedCrop", size=224),
    dict(type="ColorJitter", rand_apply=1.0, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    dict(type="RandomGrayscale", p=0.2),
    dict(type="GaussianBlur", rand_apply=0.1, sigma_min=0.1, sigma_max=2.0),
    dict(type="Solarization", rand_apply=0.2),
    dict(type="ToTensor"),
    dict(type="RandomHorizontalFlip"),
    dict(type="Normalize", **img_norm_cfg)
]]

transform_test = [
    dict(type="Resize", size=256),
    dict(type="CenterCrop", size=224),
    dict(type="ToTensor"),
    dict(type="Normalize", **img_norm_cfg)
]

batch_ratio = 4
dataset = dict(
    type='ImageNet',
    train_root='dataset/imagenet/train',
    train_list='imagenet_label/train.txt',
    test_root='dataset/imagenet/val',
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
optimizer = dict(type='LARS', lr=0.2 * batch_ratio * update_interval, weight_decay=1.5e-6, momentum=0.9,
            param_group=[
                dict(name='(bn|gn|norm)(\d+)?.(weight|bias)', weight_decay=0., lars_exclude=True, lr_ratio=0.025),
                dict(name='bias', weight_decay=0., lars_exclude=True, lr_ratio=0.025)
            ])
lr_config = dict(type='iter', policy='cosinewarmup', T_max=total_epochs, warmup_iters=13000, warmup_ratio=0.0001)

pretrained = None

Model = dict(
    type='SSModel', # Find 'SSModel' in modules/structure folder
    base_momentum=0.996,
    end_momentum=1.0
)

Evaluate="KNN" # Find 'KNN' in modules/apis/val.py:__NAME2CLS

embed_dim=2048

backbone=dict(
        type='ResNet', # Find 'ResNet' in modules/backbone folder
        depth=50,
        zero_init_residual=True,
        initial=dict(distribution='uniform', mode="fan_out", nonlinearity="relu"),
)

head=dict(
    type="BarlowTwinsHead", # Find 'BarlowTwinsHead' in modules/head folder
    in_channels=embed_dim, 
    out_channels=8096, 
    proj_layers=3, 
    lambd=0.005,
    initial=dict(nonlinearity="reset"),
)

knn=dict(    
    l2norm=True,
    topk=5
)

logger = dict(interval=100) # Details can be found in modules/utils/logger.py
saver = dict(interval=10, saveType="epoch", maxsize=4) # Details can be found in modules/utils/saver.py
evaluate_interval = 1
