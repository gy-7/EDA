import json
import os
from PIL import Image

number = 50

root_path = os.getcwd()
train_ann_path = os.path.join(root_path, 'annotations', 'instances_train2017.json')
original_img_path = os.path.join(root_path, 'train2017', '1.jpg', )
img = Image.open(original_img_path, )
w, h = img.size[0], img.size[1]

train_data = json.load(open(train_ann_path, 'r'))

test_img_dir = os.path.join(root_path, 'test2017')
test_json_path = os.path.join(root_path, 'annotations', 'new_test.json')
os.mkdir(test_img_dir)

for i in range(number):
    new_path = os.path.join(test_img_dir, str(i) + '.jpg')
    os.system(f'copy {original_img_path} {new_path}')  # windows
    # os.system(f'cp {original_img_path} {new_path}')  # linux

train_data['images'] = []
train_data['annotations'] = []

for i in range(number):
    train_data['images'].append({
        "height": h,
        "width": w,
        "id": i,
        "file_name": str(i) + ".jpg"
    })

json.dump(train_data, open(test_json_path, 'w'))
