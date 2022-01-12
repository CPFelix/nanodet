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

        # 增加mosaic数据增强
        if self.load_mosaic and self.mode == "train":
            img4, labels4, bbox4 = load_mosaic(self, idx)
            meta['img_info']['height'] = img4.shape[0]
            meta['img_info']['width'] = img4.shape[1]
            meta['img'] = img4
            meta['gt_labels'] = labels4
            meta['gt_bboxes'] = bbox4

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
    return im, img.shape[:2], im.shape[:2]  # im, hw_original, hw_resized

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
    print(labels4)
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

    # img4_draw = img4.copy()
    # for i in range(labels4.shape[0]):
    #     cv2.rectangle(img4_draw, (int(labels4[i][1]), int(labels4[i][2])), (int(labels4[i][3]), int(labels4[i][4])), (0, 0, 255), 2)
    # cv2.imwrite("/home/chenpengfei/temp/img4_draw.jpg", img4_draw)

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
    bbox4 = labels4[:, 1:]
    labels4 = labels4[:, 0]
    return img4, labels4, bbox4
