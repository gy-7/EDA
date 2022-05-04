from matplotlib import scale
import matplotlib.pyplot as plt
import mmcv 
from mmdet.datasets import PIPELINES 
 
# # 示例策略，包含两个子策略组，每个子策略组包含两个数据增强方法 
# # 为了便于演示，我们设所有的概率都为 1.0 
# demo_policies = [ 
#     [ 
#         # dict(type='Posterize', bits=4, prob=1.),  # 降低图片位数 
#         dict(type='Rotate', level=1., prob=1.,scale=1, max_rotate_angle=90)   # 旋转 
#     ], 
#     [ 
#         # dict(type='Solarize', thr=256 / 9 * 4, prob=1.),  # 翻转部分暗色像素 
#         # dict(type='AutoContrast', prob=1.)                # 自动调整对比度 
#         dict(type='Rotate', level=1., prob=1.,scale=2, max_rotate_angle=90)   # 旋转 
#     ], 
# ] 
# # 数据增强配置，利用 Registry 机制创建数据增强对象 
# auto_aug_cfg = dict( 
#     type='AutoAugment', 
#     policies=demo_policies, 
#     # hparams=dict(pad_val=0),  # 设定一些所有子策略共用的参数，如填充值（pad_val） 
# ) 



# 增加以下代码
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=135,
        interpolation=1,
        p=1.),
    dict(type='RandomBrightnessContrast', brightness_limit=[
         0.1, 0.3], contrast_limit=[0.1, 0.3], p=0.2),
    dict(type='OneOf',
         transforms=[dict(type='RGBShift', r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
                     dict(type='HueSaturationValue', hue_shift_limit=20,
                          sat_shift_limit=30, val_shift_limit=20, p=1.0)
                     ], p=0.3), 
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.2),
    dict(type='OneOf', transforms=[
        dict(type='Blur', blur_limit=3, p=1.0),
        dict(type='MedianBlur', blur_limit=3, p=1.0)], p=0.1)
]

albu_aug_cfg= dict(
        type='Albu',
        transforms=albu_train_transforms,
        # bbox_params=dict(
        #     type='BboxParams',
        #     format='pascal_voc',
        #     label_fields=['gt_labels'],
        #     min_visibility=0.0,
        #     filter_lost_elements=True),
        # keymap={
        #     'img': 'image',
        #     'gt_bboxes': 'bboxes'
        # },
        # update_pad_shape=False,
        # skip_img_without_anno=True
        )

while True:
     aug = PIPELINES.build(albu_aug_cfg) 
     
     img = mmcv.imread("./demo/clarins.jpg") 
     # 为了便于信息在预处理函数之间传递，数据增强项的输入和输出都是字典 
     img_info = {'img': img} 
     img_aug = aug(img_info)['img'] 
     
     # mmcv.imshow(img_aug) 
     plt.grid()
     plt.imshow(img_aug)
     plt.show()
