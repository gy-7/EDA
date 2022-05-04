import json
import os
import random

workdir = os.path.join(os.getcwd())


def sub_solve():
    json_file = os.path.join(workdir, 'eva_bbox.json')
    data = json.load(open(json_file, 'r', encoding='utf-8'))

    imgid2category = {}
    imgid2score = {}
    imgid2bboxnum = {}
    for i in data:
        imgid = i['image_id']
        imgcategory = i['category_id']
        imgscore = i['score']
        if imgid not in imgid2category:
            imgid2category[imgid] = imgcategory
            imgid2score[imgid] = imgscore
            imgid2bboxnum[imgid] = 1
        else:
            imgid2bboxnum[imgid] += 1
            if imgscore > imgid2score[imgid]:
                imgid2category[imgid] = imgcategory
                imgid2score[imgid] = imgscore


    i=0
    while i < len(data):
    # for i in range(len(data)):
        if data[i]['score'] < 0.25:
            data.pop(i)
            continue

        # imgid = data[i]['image_id']
        # if imgid2bboxnum[imgid] == 1:
        #     data[i]['score'] = random.randint(95,100)*0.01
        #     continue
        #
        # if data[i]['score'] < 0.5:
        #     if imgid2category[imgid] == data[i]['category_id']:
        #         data[i]['score'] = 0.7 + 0.1 * data[i]['score']  # 分低同类的
        #     else:
        #         data[i]['score'] = 0.6 + 0.1 * data[i]['score']  # 分低不同类的
        # elif imgid2category[imgid] != data[i]['category_id']:
        #     data[i]['score'] = 0.5 + 0.1 * data[i]['score']  # 分高不同类的
        #
        # data[i]['category_id'] = imgid2category[imgid]
        i+=1

    with open(os.path.join(workdir, 'new_eva_bbox.json'), "w") as jsonFile:
        json.dump(data, jsonFile)


if __name__ == '__main__':
    sub_solve()
    # classes=[str(i+1) for i in range(50)]
    # print(classes)
