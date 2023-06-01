# selfsupervised
Code repository for double blind.

## Installation

### Requirements

- Python 3.6+
- PyTorch 1.6+
- [mmcv](https://github.com/open-mmlab/mmcv) 0.6.0+

To config environment, one can run:
```
conda create -n torch1.6 python=3.6.7
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install mmcv==0.6.0
```
Notice that you should install a compatible version of PyTorch with your Cuda version (here we use **cudatoolkit=10.1**). Please refer to [pytorch](https://pytorch.org/get-started/locally/) to find a detailed installation for PyTorch.

## Getting Started
Before running a scipt, you would better run 
```
ln -s ${DATASET_ROOT} dataset
```
to configure your data path. If your folder structure is different, you may need to change the corresponding paths in config files.
```
selfsupervised
├── configs
├── imagenet_label
├── modules
├── dataset
│   ├── imagenet
│   │   ├── train
│   │   ├── val
│   ├── cifar
│   │   ├── cifar-10-batches-py
```
We provide a [task.sh](task.sh) and some [configs](configs) to train models. Detailed information can be found in [docs/task.md](docs/task.md)
In [sstrain.sh](sstrain.sh), one can run:
``` bash
runModel ${configfile} ${logname}
````
For example, to run our model, one can replace the command in [bin/sstrain.sh](sstrain.sh) with
``` bash
runModel interclass log1
```
The above command will use **configs/imagenette/icc.py** as the configuration, and output training logs and checkpoints in **result/log1**.