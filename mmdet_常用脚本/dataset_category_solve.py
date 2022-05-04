# 将数据集中类别名改为 1，2，3，，，，50
# 备份好原先的json文件，然后转变类别名。
import json
import os

workdir = os.path.join(os.getcwd(),'train','annotations')

def dataset_solve():
    json_file = os.path.join(workdir, 'instances_train2017.json')
    data = json.load(open(json_file, 'r',encoding='utf-8'))

    for i in range(len(data['categories'])):
        data['categories'][i]['name']=str(i+1)
        # print('"',data['categories'][i]['name'],'",')

    with open(os.path.join(workdir, 'instances_train2017_solve_category.json'), "w") as jsonFile:
        json.dump(data, jsonFile)

if __name__ == '__main__':
    dataset_solve()
    # classes=[str(i+1) for i in range(50)]
    # print(classes)

