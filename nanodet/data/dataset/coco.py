# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO

from .base import BaseDataset

import random


class CocoDataset(BaseDataset):
    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'license': 2,
          'file_name': '000000000139.jpg',
          'coco_url': 'http://images.cocodataset.org/val2017/000000000139.jpg',
          'height': 426,
          'width': 640,
          'date_captured': '2013-11-21 01:34:01',
          'flickr_url':
              'http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg',
          'id': 139},
         ...
        ]
        """
        self.coco_api = COCO(ann_path)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)

        # 增加mosaic数据增强
        if (isinstance(self.input_size, int)):
            self.mosaic_border = [-self.input_size // 2, -self.input_size // 2]
        else:
            self.mosaic_border = [-self.input_size[1] // 2, -self.input_size[0] // 2]  # 注意按照[H, W]格式，否则random_perspective函数会出现异常。
        self.indices = range(len(self.img_ids))

        return img_info

    def get_per_img_info(self, idx):
        img_info = self.data_info[idx]
        file_name = img_info["file_name"]
        height = img_info["height"]
        width = img_info["width"]
        id = img_info["id"]
        if not isinstance(id, int):
            raise TypeError("Image id must be int.")
        info = {"file_name": file_name, "height": height, "width": width, "id": id}
        return info

    def get_img_annotation(self, idx):
        """
        load per image annotation
        :param idx: index in dataloader
        :return: annotation dict
        """
        img_id = self.img_ids[idx]
        ann_ids = self.coco_api.getAnnIds([img_id])
        anns = self.coco_api.loadAnns(ann_ids)
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        if self.use_instance_mask:
            gt_masks = []
        if self.use_keypoint:
            gt_keypoints = []
        for ann in anns:
            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            # 转化为x1 y1 x2 y2
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get("iscrowd", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann["category_id"]])
                if self.use_instance_mask:
                    gt_masks.append(self.coco_api.annToMask(ann))
                if self.use_keypoint:
                    gt_keypoints.append(ann["keypoints"])
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        annotation = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore
        )
        if self.use_instance_mask:
            annotation["masks"] = gt_masks
        if self.use_keypoint:
            if gt_keypoints:
                annotation["keypoints"] = np.array(gt_keypoints, dtype=np.float32)
            else:
                annotation["keypoints"] = np.zeros((0, 51), dtype=np.float32)
        return annotation

    def get_train_data(self, idx):
        """
        Load image and annotation
        :param idx:
        :return: meta-data (a dict containing image, annotation and other information)
        """
        img_info = self.get_per_img_info(idx)
        file_name = img_info["file_name"]
        image_path = os.path.join(self.img_path, file_name)
        img = cv2.imread(image_path)
        if img is None:
            print("image {} read failed.".format(image_path))
            raise FileNotFoundError("Cant load image! Please check image path!")
        ann = self.get_img_annotation(idx)
        meta = dict(
            img=img, img_info=img_info, gt_bboxes=ann["bboxes"], gt_labels=ann["labels"]
        )
        if self.use_instance_mask:
            meta["gt_masks"] = ann["masks"]
        if self.use_keypoint:
            meta["gt_keypoints"] = ann["keypoints"]

        input_size = self.input_size
        if self.multi_scale:
            input_size = self.get_random_size(self.multi_scale, input_size)

        # 增加mosaic数据增强，并设置概率
        if ((random.random() < self.load_mosaic) and (self.mode == "train")):
            img4, labels4, bbox4 = load_mosaic(self, idx)
            meta['img_info']['height'] = img4.shape[0]
            meta['img_info']['width'] = img4.shape[1]
            meta['img'] = img4
            meta['gt_labels'] = labels4
            meta['gt_bboxes'] = bbox4

        # 增加cut_mosaic数据增强，并设置概率
        if ((random.random() < self.cut_mosaic) and (self.mode == "train")):
            img4, labels4, bbox4 = cut_mosaic(self, idx)
            meta['img_info']['height'] = img4.shape[0]
            meta['img_info']['width'] = img4.shape[1]
            meta['img'] = img4
            meta['gt_labels'] = labels4
            meta['gt_bboxes'] = bbox4

        # 对香烟区域进行裁切（解决香烟目标过小问题）
        if ((random.random() < self.crop_cigarette) and (self.mode == "train") and (2 in ann["labels"])):
            img4, labels4, bbox4 = cut_cigarette(self, idx)
            meta['img_info']['height'] = img4.shape[0]
            meta['img_info']['width'] = img4.shape[1]
            meta['img'] = img4
            meta['gt_labels'] = labels4
            meta['gt_bboxes'] = bbox4

            # #保存预处理后的图片和对应标注
            # img_draw = meta["img"].copy()
            # for i, box in enumerate(meta["gt_bboxes"]):
            #     cv2.rectangle(img_draw, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            # imgname = "img_draw_" + str(random.randint(0, 10000)) + ".jpg"
            # savePath = "/home/chenpengfei/temp/nanodet_20220222/"
            # if not os.path.exists(savePath):
            #     os.makedirs(savePath)
            # cv2.imwrite(savePath + imgname, img_draw)
            # print("OK")

        meta = self.pipeline(self, meta, input_size)

        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))

        return meta

    def get_val_data(self, idx):
        """
        Currently no difference from get_train_data.
        Not support TTA(testing time augmentation) yet.
        :param idx:
        :return:
        """
        # TODO: support TTA
        return self.get_train_data(idx)

# 增加mosaic数据增强
def map_newsize(x, h0, w0, w, h, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] * w / w0 + padw  # top left x
    y[:, 1] = x[:, 1] * h / h0 + padh  # top left y
    y[:, 2] = x[:, 2] * w / w0 + padw  # bottom right x
    y[:, 3] = x[:, 3] * h / h0 + padh  # bottom right y
    return y

def load_image(self, i):
    img_info = self.get_per_img_info(i)
    file_name = img_info["file_name"]
    image_path = os.path.join(self.img_path, file_name)
    img = cv2.imread(image_path)
    h0, w0 = img.shape[:2]
    if (isinstance(self.input_size, int)):
        r = self.input_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
    else:
        r = max(self.input_size[0], self.input_size[1]) / max(h0, w0)  # ratio
        im = cv2.resize(img, (self.input_size[0], self.input_size[1]),
                        interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    return im, img,  img.shape[:2], im.shape[:2]  # im, im_original, hw_original, hw_resized

def load_mosaic(self, idx):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    s = self.input_size
    if (isinstance(s, int)):
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    else:
        yc, _, _, xc = [int(random.uniform(-x, 2 * y + x)) for x in self.mosaic_border for y in [s[1], s[0]]]
        # xc = int(random.uniform(-self.mosaic_border[0], 2 * s[0] + self.mosaic_border[0]))
        # yc = int(random.uniform(-self.mosaic_border[1], 2 * s[1] + self.mosaic_border[1]))
    # print(yc, xc)
    indices = [idx] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, (h0, w0), (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            if (isinstance(s, int)):
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            else:
                img4 = np.full((s[1] * 2, s[0] * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            if (isinstance(s, int)):
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            else:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s[0] * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            if (isinstance(s, int)):
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            else:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s[1] * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            if (isinstance(s, int)):
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s[0] * 2), min(s[1] * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        # print("\n----------------------------\n")
        # print(i)
        # print(img4.shape)
        # print(img.shape)
        # print(y1a, y2a, x1a, x2a, y1b, y2b, x1b, x2b)
        # print(img4[y1a:y2a, x1a:x2a].shape)
        # print(img[y1b:y2b, x1b:x2b].shape)
        # print("----------------------------\n")
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        # cv2.imwrite("/home/chenpengfei/temp/" + str(i) + ".jpg", img4)
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels = []
        ann = self.get_img_annotation(index).copy()
        for i,box in enumerate(ann["bboxes"]):
            label = np.append(ann["labels"][i], box)
            labels.append(label)
        labels = np.array(labels)
        if labels.size > 0:
            labels[:, 1:] = map_newsize(labels[:, 1:], h0, w0, w, h, padw, padh)  # normalized xywh to pixel xyxy format
        else:
            continue  # 如果没有标注信息则保存在labels4，不然之后concatenate会报错
        labels4.append(labels)

    # Concat/clip labels
    # print(labels4)
    labels4 = np.concatenate(labels4, 0)
    if (isinstance(s, int)):
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    else:
        x_index = [1, 3]
        y_index = [2, 4]
        for x in (labels4[:, x_index], *segments4):
            labels4[:, x_index] = np.clip(x, 0, 2 * s[0], out=x)  # clip when using random_perspective()
        for y in (labels4[:, y_index], *segments4):
            labels4[:, y_index] = np.clip(y, 0, 2 * s[1], out=y)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    img4_draw = img4.copy()
    saveDir = "/home/chenpengfei/temp/nanodet_mosaic/"
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    for i in range(labels4.shape[0]):
        if (int(labels4[i][0]) == 0):
            cv2.rectangle(img4_draw, (int(labels4[i][1]), int(labels4[i][2])), (int(labels4[i][3]), int(labels4[i][4])),
                          (0, 0, 255), 2)
        else:
            cv2.rectangle(img4_draw, (int(labels4[i][1]), int(labels4[i][2])), (int(labels4[i][3]), int(labels4[i][4])),
                          (255, 0, 0), 2)
    imagename = "img4_draw_" + str(random.randint(0, 10000)) + ".jpg"
    cv2.imwrite(saveDir + imagename, img4_draw)

    # Augment
    # img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
    #
    # img4, labels4 = random_perspective(img4, labels4, segments4,
    #                                    degrees=self.hyp['degrees'],
    #                                    translate=self.hyp['translate'],
    #                                    scale=self.hyp['scale'],
    #                                    shear=self.hyp['shear'],
    #                                    perspective=self.hyp['perspective'],
    #                                    border=self.mosaic_border)  # border to remove

    # img4_draw = img4.copy()
    # for i in range(labels4.shape[0]):
    #     cv2.rectangle(img4_draw, (int(labels4[i][1]), int(labels4[i][2])), (int(labels4[i][3]), int(labels4[i][4])), (0, 0, 255), 2)
    # cv2.imwrite("/home/chenpengfei/temp/img4_draw.jpg", img4_draw)
    bbox4 = labels4[:, 1:].astype(np.float32)
    labels4 = labels4[:, 0].astype(np.int64)
    return img4, labels4, bbox4

# 判断坐标框大小是否符合要求
def smallBox(x, pixels):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    w = y[2] - y[0]
    h = y[3] - y[1]
    if ((w < pixels) or (h < pixels)):
        return False
    else:
        return True


# 变形版Masoic，每张图裁出中间区域四分之一进行拼接，相比masoic不会减小目标尺寸
def cut_mosaic(self, idx):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    s = self.input_size
    if (isinstance(s, int)):
        yc, xc = [int(random.uniform(0, s / 2)), int(random.uniform(0, s / 2))]  # mosaic center x, y
    else:
        yc, xc = [int(random.uniform(0, s[1] / 2)), int(random.uniform(0, s[0] / 2))]
        # xc = int(random.uniform(-self.mosaic_border[0], 2 * s[0] + self.mosaic_border[0]))
        # yc = int(random.uniform(-self.mosaic_border[1], 2 * s[1] + self.mosaic_border[1]))
    # print(yc, xc)
    indices = [idx] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, (h0, w0), (h, w) = load_image(self, index)

        part_w = int(w / 2)
        part_h = int(h / 2)
        # place img in img4
        if i == 0:  # top left
            if (isinstance(s, int)):
                img4 = np.full((s, s, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            else:
                img4 = np.full((s[1], s[0], img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = 0, 0, part_w, part_h  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = xc, yc, xc + part_w, yc + part_h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = part_w, 0, w, part_h
            x1b, y1b, x2b, y2b = xc, yc, xc + part_w, yc + part_h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = 0, part_h, part_w, h
            x1b, y1b, x2b, y2b = xc, yc, xc + part_w, yc + part_h
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = part_w, part_h, w, h
            x1b, y1b, x2b, y2b = xc, yc, xc + part_w, yc + part_h
        # print("\n----------------------------\n")
        # print(i)
        # print(img4.shape)
        # print(img.shape)
        # print(y1a, y2a, x1a, x2a, y1b, y2b, x1b, x2b)
        # print(img4[y1a:y2a, x1a:x2a].shape)
        # print(img[y1b:y2b, x1b:x2b].shape)
        # print("----------------------------\n")
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        # cv2.imwrite("/home/chenpengfei/temp/" + str(i) + ".jpg", img4)
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels = []
        ann = self.get_img_annotation(index).copy()
        for i,box in enumerate(ann["bboxes"]):
            label = np.append(ann["labels"][i], box)
            labels.append(label)
        labels = np.array(labels)
        if labels.size > 0:
            labels[:, 1:] = map_newsize(labels[:, 1:], h0, w0, w, h, padw, padh)  # normalized xywh to pixel xyxy format
            # 坐标截断，保证不越界
            x_index = [1, 3]
            y_index = [2, 4]
            for index,x in enumerate(labels[:, x_index]):
                labels[index, x_index] = np.clip(x, x1a, x2a, out=x)  # clip when using random_perspective()
            for index,y in enumerate(labels[:, y_index]):
                labels[index, y_index] = np.clip(y, y1a, y2a, out=y)  # clip when using random_perspective()

        else:
            continue  # 如果没有标注信息则保存在labels4，不然之后concatenate会报错
        # 增加截断后坐标的大小判断，过小的标注框过滤掉
        pixels_hand = 10
        pixels_cigarette = 2
        delete_id = []
        for row in range(labels.shape[0]):
            if (labels[row][0] == 0):
                if not smallBox(labels[row][1:], pixels_hand):
                    delete_id.append(row)
            else:
                if not smallBox(labels[row][1:], pixels_cigarette):
                    delete_id.append(row)
        labels = np.delete(labels, delete_id, axis=0)

        labels4.append(labels)

    # Concat/clip labels
    # print(labels4)
    labels4 = np.concatenate(labels4, 0)
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # img4_draw = img4.copy()
    # saveDir = "/home/chenpengfei/temp/nanodet_cut_mosaic/"
    # if not os.path.exists(saveDir):
    #     os.makedirs(saveDir)
    # for i in range(labels4.shape[0]):
    #     if (int(labels4[i][0]) == 0):
    #         cv2.rectangle(img4_draw, (int(labels4[i][1]), int(labels4[i][2])), (int(labels4[i][3]), int(labels4[i][4])), (0, 0, 255), 2)
    #     else:
    #         cv2.rectangle(img4_draw, (int(labels4[i][1]), int(labels4[i][2])), (int(labels4[i][3]), int(labels4[i][4])),
    #                       (255, 0, 0), 2)
    # imagename = "img4_draw_" + str(random.randint(0, 10000)) + ".jpg"
    # cv2.imwrite(saveDir + imagename, img4_draw)

    # Augment
    # img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
    #
    # img4, labels4 = random_perspective(img4, labels4, segments4,
    #                                    degrees=self.hyp['degrees'],
    #                                    translate=self.hyp['translate'],
    #                                    scale=self.hyp['scale'],
    #                                    shear=self.hyp['shear'],
    #                                    perspective=self.hyp['perspective'],
    #                                    border=self.mosaic_border)  # border to remove

    # img4_draw = img4.copy()
    # for i in range(labels4.shape[0]):
    #     cv2.rectangle(img4_draw, (int(labels4[i][1]), int(labels4[i][2])), (int(labels4[i][3]), int(labels4[i][4])), (0, 0, 255), 2)
    # cv2.imwrite("/home/chenpengfei/temp/img4_draw.jpg", img4_draw)
    bbox4 = labels4[:, 1:].astype(np.float32)
    labels4 = labels4[:, 0].astype(np.int64)
    return img4, labels4, bbox4


# 计算IOU
def bb_intersection_over_union(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

# 将香烟区域进行一定比例裁切
def cut_cigarette(self, idx):
    labels, segments = [], []
    # Load image
    img, img_orgin, (h0, w0), (h, w) = load_image(self, idx)

    # Labels
    labels = []
    ann = self.get_img_annotation(idx).copy()

    iou_boxes = []
    iou_labels = []
    min_box = [0.0, 0.0, 0.0, 0.0]
    for i, box in enumerate(ann["bboxes"]):
        if ann["labels"][i] == 2:
            iou_labels.append(ann["labels"][i].copy())
            iou_boxes.append(ann["bboxes"][i].copy())
            min_box = ann["bboxes"][i]

    no_iou_boxes = []
    no_iou_labels = []
    for i, box in enumerate(ann["bboxes"]):
        if ann["labels"][i] != 2:
            iou = bb_intersection_over_union(min_box, box)
            if iou > 0:
                iou_labels.append(ann["labels"][i].copy())
                iou_boxes.append(ann["bboxes"][i].copy())
                min_box[0] = min(min_box[0], box[0])
                min_box[1] = min(min_box[1], box[1])
                min_box[2] = max(min_box[2], box[2])
                min_box[3] = max(min_box[3], box[3])
            else:
                no_iou_labels.append(ann["labels"][i].copy())
                no_iou_boxes.append(ann["bboxes"][i].copy())

    crop_w = w0 / 2
    crop_h = h0 / 2
    center = [int((min_box[0] + min_box[2]) / 2), int((min_box[1] + min_box[3]) / 2)]

    crop_box = [0.0, 0.0, 0.0, 0.0]
    crop_box[0] = int(center[0] - crop_w / 2)
    crop_box[1] = int(center[1] - crop_h / 2)
    crop_box[2] = int(center[0] + crop_w / 2)
    crop_box[3] = int(center[1] + crop_h / 2)

    # 针对标注框最小外接框截断crop_box
    if crop_box[0] > min_box[0]:
        crop_box[0] = min_box[0]
    if crop_box[1] > min_box[1]:
        crop_box[1] = min_box[1]
    if crop_box[2] < min_box[2]:
        crop_box[2] = min_box[2]
    if crop_box[3] < min_box[3]:
        crop_box[3] = min_box[3]
    # 针对原始图片尺寸截断crop_box
    if crop_box[0] < 0:
        crop_box[0] = 0
    if crop_box[1] < 0:
        crop_box[1] = 0
    if crop_box[2] > w0:
        crop_box[2] = w0
    if crop_box[3] > h0:
        crop_box[3] = h0

    crop_box = [int(i) for i in crop_box]
    img_crop = img_orgin[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

    for i, box in enumerate(iou_boxes):
        box1 = [0.0, 0.0, 0.0, 0.0]
        box1[0] = box[0] - crop_box[0]
        box1[1] = box[1] - crop_box[1]
        box1[2] = box[2] - crop_box[0]
        box1[3] = box[3] - crop_box[1]
        label = np.append(iou_labels[i], box1)
        labels.append(label)
    for i, box in enumerate(no_iou_boxes):
        iou = bb_intersection_over_union(crop_box, box)
        if iou > 0:
            if box[0] < crop_box[0]:
                box[0] = crop_box[0]
            if box[1] < crop_box[1]:
                box[1] = crop_box[1]
            if box[2] > crop_box[2]:
                box[2] = crop_box[2]
            if box[3] > crop_box[3]:
                box[3] = crop_box[3]

        # 截断后过小的框过滤掉
        pixels_hand = 15
        if not smallBox(box, pixels_hand):
            continue
        box1 = [0.0, 0.0, 0.0, 0.0]
        box1[0] = box[0] - crop_box[0]
        box1[1] = box[1] - crop_box[1]
        box1[2] = box[2] - crop_box[0]
        box1[3] = box[3] - crop_box[1]
        label = np.append(no_iou_labels[i], box1)
        labels.append(label)
    labels = np.array(labels)
    bboxs_crop = labels[:, 1:].astype(np.float32)
    labels_crop = labels[:, 0].astype(np.int64)
    return img_crop, labels_crop, bboxs_crop