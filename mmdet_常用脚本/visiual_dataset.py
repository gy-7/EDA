import json
import os
import cv2

workdir = os.path.join(os.getcwd(),'fewshotlogodetection_round1_train_202204','train')
images_dir=os.path.join(workdir,'images')

# 检查转变后coco的json文件，坐标是否正确。
def visiual_gt():
    json_file = os.path.join(workdir, 'instances_train2017.json')
    data = json.load(open(json_file, 'r',encoding='utf-8'))
    images=data['images']
    annotations = data['annotations']

    imageid_2_bbox={}       # {imageid:[bbox1,bbox2,,]}
    for i in annotations:
        if i['image_id'] not in imageid_2_bbox:
            imageid_2_bbox[i['image_id']]=[]
        imageid_2_bbox[i['image_id']].append(i['bbox'])

    for i in images:
        img=cv2.imread((os.path.join(images_dir,i['file_name'])))
        img_id=i['id']
        bboxes = imageid_2_bbox[img_id]

        # 生成锚框
        for bbox in bboxes:
            left_top = (int(bbox[0]), int(bbox[1]))  # 这里数据集中bbox的含义是，左上角坐标和右下角坐标。
            right_bottom = (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3]))  # 根据不同数据集中bbox的含义，进行修改。
            cv2.rectangle(img, left_top, right_bottom, (0, 255, 0), 2)  # 图像，左上角，右下坐标，颜色，粗细

        cv2.imshow('image', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    visiual_gt()
