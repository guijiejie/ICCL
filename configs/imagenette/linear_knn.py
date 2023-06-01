num_class = 10

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = [
    dict(type="Resize", size=160),
    dict(type="CenterCrop", size=128),
    dict(type="ToTensor"),
    dict(type="Normalize", **img_norm_cfg)
]
transform_test = [
    dict(type="Resize", size=160),
    dict(type="CenterCrop", size=128),
    dict(type="ToTensor"),
    dict(type="Normalize", **img_norm_cfg)
]
batch_ratio=1
dataset = dict(
    type='ImageNette',
    root="/mnt/ceph/home/yuvalliu/dataset/imagenette/imagenette2-160",
    batchsize=256 * batch_ratio,
    num_workers=32,
    trainmode="linear", 
    transform_train=transform_train, 
    testmode="linear",
    transform_test=transform_test)

Model=dict(
    type="Model"
)
Evaluate="EvalKNN"

embed_dim = 512
backbone=dict(
        type='ResNet',
        depth=18,
        zero_init_residual=False,
        frozen=True)
# embed_dim = 384
# backbone=dict(
#     type='ViT',
#     image_size=128, 
#     patch_size=16, 
#     dim=384,
#     frozen=True,
# )