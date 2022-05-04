import json
import os
import cv2

root_path = os.getcwd()
work_dir=os.path.join(root_path)

images_dir = os.path.join(work_dir, 'data','val','images')
val_json_file = os.path.join(work_dir, 'data','val','annotations','new_test.json')
sub_json_file = os.path.join(root_path, 'new_eva_bbox.json')
val_data = json.load(open(val_json_file, 'r', encoding='utf-8'))
sub_data = json.load(open(sub_json_file, 'r', encoding='utf-8'))


# 检查转变后coco的json文件，坐标是否正确。
def visiual_sub():
    imgid_2_bbox = {}  # {imageid:[bbox1,bbox2,,]}
    for i in sub_data:
        if i['image_id'] not in imgid_2_bbox:
            imgid_2_bbox[i['image_id']] = []
        imgid_2_bbox[i['image_id']].append(i['bbox'])

    for i in val_data['images']:
        img = cv2.imread(os.path.join(images_dir, i['file_name']))
        bboxes = imgid_2_bbox[i['id']]

        # 生成锚框
        for bbox in bboxes:
            left_top = (int(bbox[0]), int(bbox[1]))  # 这里数据集中bbox的含义是: xmin, ymin ,width, height
            right_bottom = (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3]))  # 根据不同数据集中bbox的含义，进行修改。
            cv2.rectangle(img, left_top, right_bottom, (0, 255, 0), 2)  # 图像，左上角，右下坐标，颜色，粗细

        cv2.imshow('image', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    visiual_sub()
