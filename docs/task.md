# task.sh说明

```bash
#是否在jizhi平台上启动任务，输入y/n。
run task in JIZHI (y/n)[default:y]:n

#输入使用的gpu类型，除了下面三种还有T4等
Set gpu name(M40/P40/V100)[default:M40]: V100

#输入使用的gpu数量
Set gpu nums(1-8)[default:4]: 4

#选择数据集的类型，对于视频号的图片预训练则选3
dataset[default:imagenette]:
1.imagenet
2.imagenette
select a dataset: 2

#这边的输出是configs/xxx目录下的所有config文件，对于预训练则采用multimodal.py
config file:
1.linear_probe.py
2.multimodal.py
select a config file: 2

#输入任意的log name，注意一个小时内启动的同名项目会覆盖。最后的log name为这边输入的log name+日期+小时，详情见底部
Set log name: mm_training

#选择启动的脚本，2是自监督预训练，3是针对弹性任务的自监督脚本，会检测是否有ckpt，如果有则加载启动
Start CMD List
*****************
1.train.sh(default)
2.sstrain.sh
3.sstrain_elastic.sh
4.test.sh
5.sleep.sh
6.resume_sstrain.sh
*****************
select a shell: 2

#这边默认即可
Model type(e.g., resnet/vit)[default:resnet]:

#任务启动的输出信息
Generating the training shell.

#将bin/sstrain.sh的模板填入任务信息保存在runningscripts/2021_09_15_11_22_46_sstrain.sh。
#启动脚本的粒度为秒级，不用担心覆盖。
SRC: bin/sstrain.sh => DST: runningscripts/2021_09_15_11_22_46_sstrain.sh

Copying the config file.
#将对应的config文件拷贝到runningscripts/2021_09_15_11_22_46_multimodal.py。
#这边config文件的拷贝为了防止启动任务后config文件存在修改。后续该任务的启动采用拷贝后的config文件。
SRC: configs/sph/multimodal.py => DST: runningscripts/2021_09_15_11_22_46_multimodal.py
```