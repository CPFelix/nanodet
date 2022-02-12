# -*- coding:utf-8 -*-
# !/usr/bin/env python

# import argparse
import json
# import matplotlib.pyplot as plt
# import skimage.io as io
import cv2
# from labelme import utils
import numpy as np
import glob
import PIL.Image
from PIL import Image, ImageDraw
from tqdm import tqdm
import os

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path='./train.json', label = [], image_path=""):
        '''
        :param labelme_json: 所有labelme的json文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.image_path = image_path
        # self.data_coco = {}
        # self.label = ['face', 'hand', 'cigarette', 'cellphone']
        self.label = label
        # self.label = ['cigarette']
        # self.exclude_label = ['face', 'cellphone']
        self.annID = 1
        self.height = 0
        self.width = 0

        # 统计不同类别框的标注数量
        self.stats = {}

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(tqdm(self.labelme_json)):
            # print(json_file + "\n")
            with open(json_file, 'r', encoding='utf-8') as fp:
                data = json.load(fp)  # 加载json文件
                image_name = json_file.split("\\")[-1].split(".")[0] + ".jpg"
                # 加入上层文件夹路径
                # image_name = json_file.split("\\").split(".")[0] + ".jpg"
                # self.images.append(self.image(data, num))
                self.images.append(self.image_from_json(data, num, image_name))
                for shapes in data['shapes']:
                    label = shapes['label']
                    # # 跳过特定类别
                    # if label in self.exclude_label:
                    #     continue
                    if label not in self.stats.keys():
                        self.stats[label] = 0
                    self.stats[label] += 1
                    if label not in self.label:
                        # self.categories.append(self.categorie(label))
                        # self.label.append(label)
                        # print(label + " is not in label list!")
                        continue
                    hasFlag = False
                    for categorie in self.categories:
                        if label == categorie["name"]:
                            hasFlag = True
                    if not hasFlag:
                        self.categories.append(self.categorie(label))
                    points = shapes['points']  # 这里的point是用rectangle标注得到的，只有两个点，需要转成四个点
                    # points.append([points[0][0],points[1][1]])
                    # points.append([points[1][0],points[0][1]])
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1

    def image(self, data, num):
        image = {}
        # img = utils.img_b64_to_arr(data['imageData'])  # 解析原图片数据
        # img=io.imread("F:\\阜康测试视频\\frame-16\\labelme\\test\\img\\" + data['imagePath']) # 通过图片路径打开图片
        # img = cv2.imread("F:\\阜康测试视频\\frame-16\\labelme\\test\\img\\" + data['imagePath'], 0)
        img = cv2.imdecode(np.fromfile(os.path.join(self.image_path, data['imagePath']), dtype=np.uint8), -1)
        height, width = img.shape[:2]
        img = None
        image['height'] = height
        image['width'] = width
        image['id'] = num + 1
        # image['file_name'] = data['imagePath'].split('/')[-1]
        image['file_name'] = data['imagePath']
        self.height = height
        self.width = width

        return image

    # 从Json文件中获取图片信息
    def image_from_json(self, data, num, image_name):
        image = {}
        image['height'] = data["imageHeight"]
        image['width'] = data["imageWidth"]
        image['id'] = num + 1
        # image['file_name'] = data['imagePath'].split('/')[-1]
        image['file_name'] = image_name
        self.height = data["imageHeight"]
        self.width = data["imageWidth"]

        return image

    def categorie(self, label):
        categorie = {}
        categorie['supercategory'] = 'None'
        categorie['id'] = self.label.index(label) + 1  # 0 默认为背景
        categorie['name'] = label
        return categorie

    def annotation(self, points, label, num):
        annotation = {}
        annotation['segmentation'] = [list(np.asarray(points).flatten())]
        annotation['iscrowd'] = 0
        annotation['image_id'] = num + 1
        # annotation['bbox'] = str(self.getbbox(points)) # 使用list保存json文件时报错（不知道为什么）
        # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] 使用该方式转成list
        annotation['bbox'] = list(map(float, self.getbbox(points)))
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
        # annotation['category_id'] = self.getcatid(label)
        annotation['category_id'] = self.getcatid(label)  # 注意，源代码默认为1
        annotation['id'] = self.annID
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return 1

    def getbbox(self, points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points

        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False, cls=MyEncoder)  # indent=4 更加美观显示

if __name__ == "__main__":
    labelme_json = glob.glob('E:\\pycharm-projects\\dataset\\DSM_Dataset_class4_20220211_fukang\\val\\json\\*.json')
    # labelme_json=['./Annotations/*.json']
    image_path = "E:\\pycharm-projects\\dataset\\DSM_Dataset_class4_20220211_fukang\\val\\image"
    save_json_path = 'E:\\pycharm-projects\\dataset\\DSM_Dataset_class4_20220211_fukang\\val\\val.json'
    object = labelme2coco(labelme_json, save_json_path, label=['face', 'hand', 'cigarette', 'cellphone'], image_path = image_path)
    print(object.stats)