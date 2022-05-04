import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

root_path = os.getcwd()
json_filepath = os.path.join(root_path, 'instances_train2017.json')
data = json.load(open(json_filepath, 'r'))
EDA_dir = './eda_visiual/'
if not os.path.exists(EDA_dir):
    # shutil.rmtree(EDA_dir)
    os.makedirs(EDA_dir)

# 准备图片数据
images = data[
    'images']  # [{"license": 0, "url": null, "file_name": "0.jpg", "height": 350, "width": 741, "date_captured": null, "id": 0}, , , , , ]
annotations = data[
    'annotations']  # [{"id": 0, "image_id": 0, "category_id": 1, "area": 17739, "bbox": [246, 77, 81, 219], "iscrowd": 0}, , , , ]

images_height = []
images_width = []
images_aspect_ratio = []  # 图片的宽高比
bboxes_height = []
bboxes_width = []
bboxes_aspect_ratio = []  # bboxes的宽高比
bboxes_num_per_image = []  # 每个图片的bbox数量

for i in images:
    images_width.append(i['width'])
    images_height.append(i['height'])
    width_height = i['width'] / i['height']
    images_aspect_ratio.append(width_height)

for i in annotations:
    bboxes_width.append(i['bbox'][2])
    bboxes_height.append(i['bbox'][3])
    width_height = i['bbox'][2] / i['bbox'][3]
    bboxes_aspect_ratio.append(width_height)

temp_num = 0
images_id = []
for i in annotations:
    if i['image_id'] not in images_id:
        images_id.append(i['image_id'])
        if temp_num > 0:
            bboxes_num_per_image.append(temp_num)
        temp_num = 1
    else:
        temp_num = temp_num + 1

# 配置绘图的参数
sns.set_style("whitegrid")

# 绘制图片宽高的分布
plt.title('Images width and height distribution')
sns.kdeplot(images_width, images_height, shade=True)
plt.savefig(EDA_dir + 'images_width_height_distribution.png', dpi=600)
plt.show()

# 绘制图片宽高比分布
plt.title('Images aspect ratio distribution')
sns.distplot(images_aspect_ratio, kde=False)
plt.savefig(EDA_dir + 'images_aspect_ratio.png', dpi=600)
plt.show()

# 绘制图片宽度比分布
plt.title('Images width distribution')
sns.distplot(images_width, kde=False)
plt.savefig(EDA_dir + 'images_width_distribution', dpi=600)
plt.show()

# 绘制图片高度比分布
plt.title('Images height distribution')
sns.distplot(images_height, kde=False)
plt.savefig(EDA_dir + 'images_height_distribution.png', dpi=600)
plt.show()

# 绘制bboxes宽高的分布
plt.title('Bboxes width and height distribution')
sns.kdeplot(bboxes_width, bboxes_height, shade=True)
plt.savefig(EDA_dir + 'bboxes_width_height_distribution.png', dpi=600)
plt.show()

# 绘制bboxes宽高比分布
plt.title('Bboxes aspect ratio distribution')
sns.distplot(bboxes_aspect_ratio, kde=False)
plt.savefig(EDA_dir + 'bboxes_aspect_ratio .png', dpi=600)
plt.show()

# 绘制bboxes宽度比分布
plt.title('Bboxes width distribution')
sns.distplot(bboxes_width, kde=False)
plt.savefig(EDA_dir + 'bboxes_width_distribution', dpi=600)
plt.show()

# 绘制bboxes高度比分布
plt.title('Bboxes height distribution')
sns.distplot(bboxes_height, kde=False)
plt.savefig(EDA_dir + 'bboxes_height_distribution.png', dpi=600)
plt.show()

# 绘制每张图片bboxes个数的分布情况
plt.title('Distribution of the number of BBoxes in each image')
sns.distplot(bboxes_num_per_image, kde=False)
plt.savefig(EDA_dir + 'bboxes_per_image_distribution.png', dpi=600)
plt.show()