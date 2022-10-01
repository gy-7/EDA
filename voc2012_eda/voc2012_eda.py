import os
import seaborn as sns
import pycocotools.coco as coco
import matplotlib.pyplot as plt

root_dir = os.getcwd()
train_ann_fp = os.path.join(root_dir, 'annotations', 'voc12_train.json')
val_ann_fp = os.path.join(root_dir, 'annotations', 'voc12_val.json')


class VOC_EDA:
    def __init__(self, json_file, type='train'):
        self.COCO_SMALL_SCALE = 32
        self.COCO_MEDIUM_SCALE = 96

        self.json_file = json_file
        voc = coco.COCO(json_file)

        self.type = type
        self.imgs = voc.dataset['images']
        self.anns = voc.dataset['annotations']
        self.cats = voc.dataset['categories']
        self.img_ids = voc.getImgIds()
        self.ann_ids = voc.getAnnIds()
        self.cat_ids = voc.getCatIds()

        self.cat2imgs = voc.catToImgs
        self.img2anns = voc.imgToAnns

        self.imgs_num = len(self.imgs)
        self.objs_num = len(self.anns)

        # data to be collected
        self.small_objs_num = 0
        self.medium_objss_num = 0
        self.large_objss_num = 0

        self.small_objs = []
        self.medium_objs = []
        self.large_objs = []

        self.cat2objs = {}
        self.small_cat2objs = {}  # small objects classes distribution
        self.medium_cat2objs = {}  # medium objects classes distribution
        self.large_cat2objs = {}  # large objects classes distribution
        self.cat2objs_num = {}  # objects classes distribution
        self.small_cat2objs_num = {}  # small objects classes distribution
        self.medium_cat2objs_num = {}  # medium objects classes distribution
        self.large_cat2objs_num = {}  # large objects classes distribution

        # plot use data
        self.catid2name = {}  # 用于绘图中显示类别名字
        self.cats_plot = []  # voc 所有尺寸目标的类别分布
        self.small_cats_plot = []  # 小目标中每个类的分布情况
        self.medium_cats_plot = []  # 中目标中每个类的分布情况
        self.large_cats_plot = []  # 大目标中每个类的分布情况

        # 每个类的小，中，大目标的数量
        self.size_distribution = {}


def collect_data(voc):
    # collect small, medium, large objects
    for ann in voc.anns:
        if ann['area'] < voc.COCO_SMALL_SCALE ** 2:
            voc.small_objs_num += 1
            voc.small_objs.append(ann)
        elif ann['area'] < voc.COCO_MEDIUM_SCALE ** 2:
            voc.medium_objs.append(ann)
            voc.medium_objss_num += 1
        else:
            voc.large_objs.append(ann)
            voc.large_objss_num += 1

    for i in voc.cat_ids:
        voc.cat2objs[i] = []
        voc.small_cat2objs[i] = []
        voc.medium_cat2objs[i] = []
        voc.large_cat2objs[i] = []
        voc.cat2objs_num[i] = 0
        voc.small_cat2objs_num[i] = 0
        voc.medium_cat2objs_num[i] = 0
        voc.large_cat2objs_num[i] = 0
        voc.size_distribution[i] = []

    for i in voc.cats:
        voc.catid2name[i['id']] = i['name']

    # collect small, medium, large class distribution
    for i in voc.anns:
        voc.cat2objs[i['category_id']].append(i)
        voc.cat2objs_num[i['category_id']] += 1
        voc.cats_plot.append(voc.catid2name[i['category_id']])
        if i['area'] < voc.COCO_SMALL_SCALE ** 2:
            voc.small_cat2objs[i['category_id']].append(i)
            voc.small_cat2objs_num[i['category_id']] += 1
            voc.small_cats_plot.append(voc.catid2name[i['category_id']])
            voc.size_distribution[i['category_id']].append('s')
        elif i['area'] < voc.COCO_MEDIUM_SCALE ** 2:
            voc.medium_cat2objs[i['category_id']].append(i)
            voc.medium_cat2objs_num[i['category_id']] += 1
            voc.medium_cats_plot.append(voc.catid2name[i['category_id']])
            voc.size_distribution[i['category_id']].append('m')
        else:
            voc.large_cat2objs[i['category_id']].append(i)
            voc.large_cat2objs_num[i['category_id']] += 1
            voc.large_cats_plot.append(voc.catid2name[i['category_id']])
            voc.size_distribution[i['category_id']].append('l')

    assert len(voc.small_objs) == voc.small_objs_num == sum(voc.small_cat2objs_num.values())
    assert len(voc.medium_objs) == voc.medium_objss_num == sum(voc.medium_cat2objs_num.values())
    assert len(voc.large_objs) == voc.large_objss_num == sum(voc.large_cat2objs_num.values())
    assert len(voc.anns) == voc.objs_num == sum(voc.cat2objs_num.values())


def plot_voc_class_distribution(plot_data, plot_order, save_fp, plot_title, plot_y_heigh,
                                plot_y_heigh_residual=[20, 20]):
    # 绘制voc数据集的类别分布
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 8))  # 图片的宽和高，单位为inch
    plt.title(plot_title, fontsize=9)  # 标题
    plt.xlabel('class', fontsize=8)  # x轴名称
    plt.ylabel('counts', fontsize=8)  # y轴名称
    plt.xticks(rotation=90, fontsize=8)  # x轴标签竖着显示
    plt.yticks(fontsize=8)
    for x, y in enumerate(plot_y_heigh):
        if 'train' in save_fp:
            plt.text(x, y + plot_y_heigh_residual[0], '%s' % y, ha='center', fontsize=7, rotation=0)
        else:
            plt.text(x, y + plot_y_heigh_residual[1], '%s' % y, ha='center', fontsize=7, rotation=0)
    ax = sns.countplot(x=plot_data, palette="PuBu_r", order=plot_order)  # 绘制直方图，palette调色板，蓝色由浅到深渐变。
    # palette样式：https://blog.csdn.net/panlb1990/article/details/103851983
    plt.savefig(os.path.join(save_fp), dpi=500)
    plt.show()


def plot_size_distribution(plot_data, save_fp, plot_title, plot_order=['s', 'm', 'l']):
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 12))  # 图片的宽和高，单位为inch
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=1, hspace=1.5)  # 调整子图间距

    for idx, size_data in enumerate(plot_data.values()):
        plt.subplot(4, 5, idx + 1)
        plt.xticks(rotation=0, fontsize=18)  # x轴标签竖着显示
        plt.yticks(fontsize=18)
        plt.xlabel('size', fontsize=20)  # x轴名称
        plt.ylabel('count', fontsize=20)  # y轴名称
        plt.title(plot_title[idx], fontsize=24)  # 标题
        sns.countplot(x=size_data, palette="PuBu_r", order=plot_order)  # 绘制直方图，palette调色板，蓝色由浅到深渐变。

    plt.savefig(save_fp, dpi=500, pad_inches=0)
    plt.show()


def run_plot_voc_class_distribution(voc, save_dir):
    # # 绘制voc数据集的类别分布
    plot_order = [i for i in voc.catid2name.values()]

    plot_heigh = [i for i in voc.cat2objs_num.values()]
    save_fp = os.path.join(save_dir, f'voc_{voc.type}_class_distribution.png')
    plot_voc_class_distribution(voc.cats_plot, plot_order, save_fp, 'VOC train2012 class distribution', plot_heigh,
                                plot_y_heigh_residual=[20, 20])

    plot_heigh = [i for i in voc.small_cat2objs_num.values()]
    save_fp = os.path.join(save_dir, f'voc_{voc.type}_small_class_distribution.png')
    plot_voc_class_distribution(voc.small_cats_plot, plot_order, save_fp, 'VOC train2012 small class distribution',
                                plot_heigh,
                                plot_y_heigh_residual=[20, 20])

    plot_heigh = [i for i in voc.medium_cat2objs_num.values()]
    save_fp = os.path.join(save_dir, f'voc_{voc.type}_medium_class_distribution.png')
    plot_voc_class_distribution(voc.medium_cats_plot, plot_order, save_fp, 'VOC train2012 medium class distribution',
                                plot_heigh, plot_y_heigh_residual=[20, 20])

    plot_heigh = [i for i in voc.large_cat2objs_num.values()]
    save_fp = os.path.join(save_dir, f'voc_{voc.type}_large_class_distribution.png')
    plot_voc_class_distribution(voc.large_cats_plot, plot_order, save_fp, 'VOC train2012 large class distribution',
                                plot_heigh,
                                plot_y_heigh_residual=[20, 20])


def run_plot_voc_size_distribution(voc, save_dir):
    # 绘制voc数据集各类别的尺寸分布
    plot_order = [i for i in voc.catid2name.values()]
    save_fp = os.path.join(save_dir, f'voc_{voc.type}_size_distribution.png')
    plot_size_distribution(voc.size_distribution, save_fp, plot_order)


if __name__ == '__main__':
    cwd = os.getcwd()
    voc_res_dir = os.path.join(cwd, "voc2012_results")
    if not os.path.exists(voc_res_dir):
        os.makedirs(voc_res_dir)

    print("analyze voc train dataset...")
    print("-" * 50)
    voc_train = VOC_EDA(train_ann_fp, type='train')
    collect_data(voc_train)
    print("voc train images num:", voc_train.imgs_num)
    print("voc train objects num:", voc_train.objs_num)
    print("voc small objects num:", voc_train.small_objs_num)
    print("voc medium objects num:", voc_train.medium_objss_num)
    print("voc large objects num:", voc_train.large_objss_num)
    print("voc small objects percent:", voc_train.small_objs_num / voc_train.objs_num)
    print("voc medium objects percent:", voc_train.medium_objss_num / voc_train.objs_num)
    print("voc large objects percent:", voc_train.large_objss_num / voc_train.objs_num)
    run_plot_voc_class_distribution(voc_train, voc_res_dir)
    run_plot_voc_size_distribution(voc_train, voc_res_dir)
    print("-" * 50)
    print()

    print("analyze voc val dataset...")
    print("-" * 50)
    voc_val = VOC_EDA(val_ann_fp, type='val')
    collect_data(voc_val)
    print("voc val images num:", voc_val.imgs_num)
    print("voc val objects num:", voc_val.objs_num)
    print("voc small objects num:", voc_val.small_objs_num)
    print("voc medium objects num:", voc_val.medium_objss_num)
    print("voc large objects num:", voc_val.large_objss_num)
    print("voc small objects percent:", voc_val.small_objs_num / voc_val.objs_num)
    print("voc medium objects percent:", voc_val.medium_objss_num / voc_val.objs_num)
    print("voc large objects percent:", voc_val.large_objss_num / voc_val.objs_num)
    run_plot_voc_class_distribution(voc_val, voc_res_dir)
    run_plot_voc_size_distribution(voc_val, voc_res_dir)
    print("-" * 50)
