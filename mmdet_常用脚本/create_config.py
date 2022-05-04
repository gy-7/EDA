# kaggle 专用
# 在mmdetection的根目录下运行，报错多那个参数，就把create_mm_config中那个参数赋值给注释掉。
import os
import random
import numpy as np
import torch
from mmcv import Config

root_path = os.getcwd()
model_name = 'cascade_mask_rcnn_r101'  # 改成自己要使用的模型名字
work_dir = os.path.join(root_path, "work_dirs",model_name)  # 训练过程中，保存文件的路径，不用动。
baseline_cfg_path = os.path.join('configs','cascade_rcnn','cascade_mask_rcnn_r101_fpn_mstrain_3x_coco.py')            # 改成自己要使用的模型的路径
save_cfg_path = os.path.join(work_dir , 'config.py')  # 生成的配置文件保存的路径


train_data_images = os.path.join(root_path,'data','train','images')# 改成自己训练集图片的目录。
val_data_images = os.path.join(root_path,'data','train','images')# 改成自己验证集图片的目录。
test_data_images = os.path.join(root_path,'data','val','images')# 改成自己测试集图片的目

train_ann_file = os.path.join(root_path,'data','train','annotations','new_train.json')# 修改为自己的数据集的训练集json
val_ann_file = os.path.join(root_path,'data','train','annotations','new_val.json')# 修改为自己的数据集的验证集json
test_ann_file = os.path.join(root_path,'data','val','annotations','new_test.json') # 修改为自己的数据集的验证集json录。

# 去找个网址里找你对应的模型的网址: https://github.com/open-mmlab/mmdetection/blob/master/README_zh-CN.md
load_from = os.path.join(work_dir,'checkpoint.pth')

# File config
num_classes = 50  # 改成自己的类别数。
classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
           '11', '12', '13', '14', '15', '16', '17', '18', '19',
           '20', '21', '22', '23', '24', '25', '26', '27', '28',
           '29', '30', '31', '32', '33', '34', '35', '36', '37',
           '38', '39', '40', '41', '42', '43', '44', '45', '46',
           '47', '48', '49', '50')                              # 改成自己的类别

# Train config              # 根据自己的需求对下面进行配置
gpu_ids = range(0,1)  # 改成自己要用的gpu
gpu_num=1
total_epochs = 20  # 改成自己想训练的总epoch数
batch_size = 2 ** 1  # 根据自己的显存，改成合适数值，建议是2的倍数。
num_worker = 1  # 比batch_size小，就行
log_interval = 300  # 日志打印的间隔
checkpoint_interval = 7  # 权重文件保存的间隔
lr = 0.02 * batch_size * gpu_num / 16  # 学习率
ratios=[0.25,0.5,1.0,2.0,4.0]
strides=[4,8,16,32,64]


cfg = Config.fromfile(baseline_cfg_path)

if not os.path.exists(work_dir):
    os.makedirs(work_dir)

cfg.work_dir = work_dir
print("Save config dir:", work_dir)

cfg.classes = classes
cfg.data.train.dataset.img_prefix = train_data_images
cfg.data.train.dataset.classes = classes
cfg.data.train.dataset.ann_file = train_ann_file
#     cfg.data.train.img_prefix = train_data_images
#     cfg.data.train.classes = classes
#     cfg.data.train.ann_file = train_ann_file

cfg.data.val.img_prefix = val_data_images
cfg.data.val.classes = classes
cfg.data.val.ann_file = val_ann_file

cfg.data.test.img_prefix = test_data_images
cfg.data.test.classes = classes
cfg.data.test.ann_file = test_ann_file

cfg.data.samples_per_gpu = batch_size  # Batch size of a single GPU used in testing
cfg.data.workers_per_gpu = num_worker  # Worker to pre-fetch data for each single GPU
cfg.log_config.interval = log_interval

for head in cfg.model.roi_head.bbox_head:                   # 修改配置文件中bbox的类别数
    head.num_classes = num_classes
if "mask_head" in cfg.model.roi_head:
    cfg.model.roi_head.mask_head.num_classes = num_classes  # 修改配置文件中mask的类别数

cfg.load_from = load_from
cfg.runner.max_epochs = total_epochs
cfg.total_epochs = total_epochs
cfg.optimizer.lr = lr
cfg.checkpoint_config.interval = checkpoint_interval
cfg.model.rpn_head.anchor_generator.ratios=ratios
cfg.model.rpn_head.anchor_generator.strides=strides

cfg.dump(save_cfg_path)
print(save_cfg_path)
print("—" * 50)
print(f'CONFIG:\n{cfg.pretty_text}')
print("—" * 50)