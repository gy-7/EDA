# 如果只是根据json划分数据集，也就是将instances_train2017.json中的东西划分为train和valid，那就用这个脚本，又快又稳。
import json
import os
import random
from pycocotools.coco import COCO

# coco.getCatIds() # 此函数用于获取加载对象所包含的所有类别的id（即category 的序号）
# coco.getAnnIds() # 获取加载对象所包含的所有标记信息（就是所有图片的Segmentation，即分割的坐标数据）
# coco.getImgIds() # 获取所有 标记所对应的原图id

root=os.getcwd()
workdir=os.path.join(root,'train','annotations')        # 把wrokdir改为包含你json文件的目录

coco = COCO(os.path.join(workdir,'instances_train2017.json'))
class_names = [coco.cats[catId]['name'] for catId in coco.getCatIds()]
categories = [dict(id=i+1, name=name) for i, name in enumerate(class_names)]

annotaions_train = []
images_train = []
annotaions_val = []
images_val = []
for catId in coco.getCatIds():
    imgIds = coco.getImgIds(catIds=[catId])
    random.shuffle(imgIds)
    for imgId in imgIds[:10]:
        img = coco.imgs[imgId]
        images_val.append(img)
        anns = coco.imgToAnns[imgId]
        for ann in anns:
            annotaions_val.append(ann)
    for imgId in imgIds[10:]:
        img = coco.imgs[imgId]
        images_train.append(img)
        anns = coco.imgToAnns[imgId]
        for ann in anns:
            annotaions_train.append(ann)
new_train = {"images": images_train, "type": "instances", "annotations": annotaions_train, "categories": categories}
new_valid = {"images": images_val, "type": "instances", "annotations": annotaions_val, "categories": categories}

with open(os.path.join(workdir, 'new_train.json'), "w") as jsonFile:
    json.dump(new_train, jsonFile)

with open(os.path.join(workdir, 'new_valid.json'), "w") as jsonFile:
    json.dump(new_valid, jsonFile)

