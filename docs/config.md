# Config文件

```python
num_class = 1000

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#创建增广操作，具体方式可参考modules/dataset/pipeline/transforms.py
#这边是对于一张图片创建了两个增广。
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
#创建dataloader，采用ImageNet的dataloader。
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

#创建optimizer，并且对于bn|gn|norm . weight|bias设置更小的学习率，取消weight decay。
optimizer = dict(type='LARS', lr=0.2 * batch_ratio * update_interval, weight_decay=1.5e-6, momentum=0.9,
            param_group=[
                dict(name='(bn|gn|norm)(\d+)?.(weight|bias)', weight_decay=0., lars_exclude=True, lr_ratio=0.025),
                dict(name='bias', weight_decay=0., lars_exclude=True, lr_ratio=0.025)
            ])

# 创建learning rate schedule。采用iter的方式更新学习率，这边也可以使用epoch。
# 更新策略采用cosine的方式，并且有13000iter的warmup。
# 对于epoch的方式如果warmup_iters=10则表明有10个epoch的wamrup。
# T_max无论是epoch的方式还是iter的方式都设置为总共epoch数即可。
lr_config = dict(type='iter', policy='cosinewarmup', T_max=total_epochs, warmup_iters=13000, warmup_ratio=0.0001)

pretrained = None

# 创建structure为SSModel
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

# 日志频率为100个iter一次。
logger = dict(interval=100) # Details can be found in modules/utils/logger.py

# 保存ckpt的频率为10个epoch一次，且最多保留4个ckpt
saver = dict(interval=10, saveType="epoch", maxsize=4) # Details can be found in modules/utils/saver.py

# 评估的频率为1个epoch一次。
evaluate_interval = 1

```