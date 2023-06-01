import torchvision
import os
import tarfile
import shutil

ARCHIVE_META = {
    'train': ('ILSVRC2012_img_train.tar', '1d675b47d978889d74fa0da5fadfb00e'),
    'val': ('ILSVRC2012_img_val.tar', '29b22e2961454d5413ddabcf34fc5622'),
    'devkit': ('ILSVRC2012_devkit_t12.tar.gz', 'fa75699e90414af021442c21a62c3abf')
}
def _is_tarxz(filename: str) -> bool:
    return filename.endswith(".tar.xz")


def _is_tar(filename: str) -> bool:
    return filename.endswith(".tar")


def _is_targz(filename: str) -> bool:
    return filename.endswith(".tar.gz")


def _is_tgz(filename: str) -> bool:
    return filename.endswith(".tgz")


def _is_gzip(filename: str) -> bool:
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename: str) -> bool:
    return filename.endswith(".zip")
    
def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)

def parse_train_archive(root, file=None, folder="train"):
    archive_meta = ARCHIVE_META["train"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]

    train_root = os.path.join(root, folder)
    extract_archive(file, train_root)

    archives = [os.path.join(train_root, archive) for archive in os.listdir(train_root)]
    for archive in archives:
        extract_archive(archive, os.path.splitext(archive)[0], remove_finished=True)

def unzipimagenet():
    if not os.path.exists("/dockerdata/ILSVRC2012_img_train.tar"):
        shutil.copy("/mnt/ceph/home/yuvalliu/dataset/ILSVRC2012_img_train.tar", "/dockerdata/ILSVRC2012_img_train.tar")
    if not os.path.exists("/dockerdata/imagenet"):
        os.mkdir("/dockerdata/imagenet")

    if not os.path.exists("/dockerdata/imagenet/train") or len([fn for fn in os.listdir("/dockerdata/imagenet/train") if 'tar' not in fn and 'zip' not in fn]) != 1000:
        parse_train_archive("/dockerdata/imagenet", "/dockerdata/ILSVRC2012_img_train.tar", folder='train')

    if not os.path.exists("/dockerdata/ILSVRC2012_img_val.tar"):
        shutil.copy("/mnt/ceph/home/yuvalliu/dataset/ILSVRC2012_img_val.tar", "/dockerdata/ILSVRC2012_img_val.tar")
    
    if not os.path.exists("/dockerdata/imagenet/val"):
        os.mkdir("/dockerdata/imagenet/val")
        os.system("tar -xf /dockerdata/ILSVRC2012_img_val.tar -C /dockerdata/imagenet/val")
    os.system("touch /dockerdata/imagenet_copy_complete.txt")