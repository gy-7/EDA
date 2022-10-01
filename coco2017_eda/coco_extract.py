import json
import os
import random
from pycocotools.coco import COCO
import shutil
from tqdm import trange

train_imgs_num = 8000  # 训练集320张，验证集80张
val_imgs_num = 1000

root = os.getcwd()
coco_origin_dir = os.path.join(root, 'coco2017')
coco_small_dir = os.path.join(root, 'coco_small')
workdir = os.path.join(coco_origin_dir, 'annotations')

if os.path.exists(coco_small_dir):
    shutil.rmtree(coco_small_dir)


def coco_extract():
    coco = COCO(os.path.join(workdir, 'instances_train2017.json'))
    categories = coco.dataset['categories']
    images_train = []
    images_val = []
    annotaions_train = []
    annotaions_val = []

    imgIds = coco.getImgIds()
    # random.shuffle(imgIds)
    for imgId in imgIds[:train_imgs_num]:
        img = coco.imgs[imgId]
        images_train.append(img)
        anns = coco.imgToAnns[imgId]
        for ann in anns:
            annotaions_train.append(ann)

    for imgId in imgIds[train_imgs_num:train_imgs_num + val_imgs_num]:
        img = coco.imgs[imgId]
        images_val.append(img)
        anns = coco.imgToAnns[imgId]
        for ann in anns:
            annotaions_val.append(ann)

    new_train = {"images": images_train, "type": "instances", "annotations": annotaions_train, "categories": categories}
    new_valid = {"images": images_val, "type": "instances", "annotations": annotaions_val, "categories": categories}

    ### save json
    os.makedirs(os.path.join(coco_small_dir, 'annotations'))

    with open(os.path.join(coco_small_dir, 'annotations', 'instances_train2017.json'), "w") as jsonFile:
        json.dump(new_train, jsonFile)

    with open(os.path.join(coco_small_dir, 'annotations', 'instances_val2017.json'), "w") as jsonFile:
        json.dump(new_valid, jsonFile)

    ### copy imgs
    imgs_train = new_train['images']
    imgs_val = new_valid['images']

    os.mkdir(os.path.join(coco_small_dir, 'train2017'))
    os.mkdir(os.path.join(coco_small_dir, 'val2017'))

    for idx in trange(len(imgs_train)):
        shutil.copy(os.path.join(coco_origin_dir, 'train2017', imgs_train[idx]['file_name']),
                    os.path.join(coco_small_dir, 'train2017'))
    for idx in trange(len(imgs_val)):
        shutil.copy(os.path.join(coco_origin_dir, 'train2017', imgs_val[idx]['file_name']),
                    os.path.join(coco_small_dir, 'val2017'))

    print("| json file saved:", os.path.join(coco_small_dir, 'annotations'))
    print("| imgs file copy complete:", coco_small_dir)
    print("| train imgs num:", len(imgs_train))
    print("| val imgs num:", len(imgs_val))
    print("| train anns num:", len(annotaions_train))
    print("| val anns num:", len(annotaions_val))
    print("| categories num:", len(categories))


if __name__ == '__main__':
    coco_extract()
