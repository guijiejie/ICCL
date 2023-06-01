# 如何创建一个新模块

具体一些备注文件可以参考[modules/head/test.py](modules/head/test.py)，[modules/structure/ImgModel.py](modules/structure/ImgModel.py).配置文件说明可参考[configs/imagenet/barlowtwins.py](configs/imagenet/barlowtwins.py). 所有支持的optimizer以及学习率调整策略参考[modules/builder.py](modules/builder.py).

## Example
以创建一个新的head为例，首先在head文件夹下创建一个新的py文件test.py
在test.py开头加入
```python
import torch.nn as nn
from .build import HEAD_REGISTERY
from .utilsfns import *

@HEAD_REGISTERY.register
class YourHead(nn.Module):
    def __init__(self, **other_params):
        pass
    def init_weights(self):
        pass
    def forward(self):
        pass
```
需保证所有模块包含这三个函数(同样适用于neck和backbone)。创建模块后会通过init_weights来初始化参数。
此外在head文件夹下的__init__.py中加入
```python
from .test import YourHead
```
之后就可以在配置文件中通过
```python
head=dict(
    type="YourHead",
    other_param1 = 1,
    other_param2 = 2,
)
```
这种方式来快捷的创建不同的head。

## Head模块与Neck模块和Backbone模块的不同
对于neck和backbone，他们的输出在structure的对应model中直接处理。如
```python
class ImgModel(nn.Module):
    def __init__(self):
        self.backbone = A()
        self.neck = B()
        self.head = C()
    
    def forward(self, x, y):
        out = self.backbone(x)
        out = self.neck(out)

        outputs = self.head(out)
        return outputs
```
其中outputs是一个字典，在train loop当中取`outputs['loss']`作为最终loss进行反向传播，outputs中的其他key是字符串，value是tensor数值。log模块会打印所有其他的key和value来帮助更好的进行日志记录。
如在head中可以通过
```python
    def forward(self, x, labels):
        loss = cross_entropy(x, labels)
        outputs['loss'] = loss # 反向传播的loss
        outputs['log1'] = torch.tensor(1) # 记录数值1，key是log1
        outputs['batchsize'] = x.size(0) # 记录数值x.size(0), key是batchsize
```
注意不能记录数组。