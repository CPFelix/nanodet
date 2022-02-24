# 获取烟遮挡数据的手部区域和对应烟坐标用于训练烟头检测
import os
import csv
from shutil import copyfile
import cv2
import numpy as np
import random
from tqdm import tqdm
import json
from PIL import Image
import math
from random import shuffle

def csv2yolov5():
    # 读取csv为yolov5标签文件
    imgPath = "E:\\pycharm-projects\\dataset\\检测数据集\\keruisite\\sigurate"
    csvfile = "E:\\pycharm-projects\\dataset\\检测数据集\\keruisite\\sigurate_smoke.csv"
    ImgsavePath = "E:\\pycharm-projects\\dataset\\DSMhand&smoke2\\temp\\keruisite\\sigurate"
    TxtsavePath = "E:\\pycharm-projects\\dataset\\DSMhand&smoke2\\temp\\keruisite\\sigurate"
    with open(csvfile, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)
            imgfile = os.path.join(imgPath, row[0])
            img = cv2.imdecode(np.fromfile(imgfile, dtype=np.uint8), 1)
            width = img.shape[1]
            height = img.shape[0]
            target_file = os.path.join(ImgsavePath, row[0])
            if not os.path.exists(ImgsavePath):
                os.makedirs(ImgsavePath)
            copyfile(imgfile, target_file)
            txtfile = os.path.join(TxtsavePath, row[0].split(".")[0] + ".txt")
            with open(txtfile, "a") as f1:
                # hand 0  smoke 1
                label = 1
                object_x_center = (int(row[1]) + int(row[3])) / 2.0 / float(width)
                object_y_center = (int(row[2]) + int(row[4])) / 2.0 / float(height)
                object_width = (int(row[3]) - int(row[1])) / float(width)
                object_height = (int(row[4]) - int(row[2])) / float(height)
                f1.write(str(label) + " " + str(object_x_center) + " " + str(object_y_center) + " " + str(object_width) + " " + str(object_height) + "\n")


def deleteNojson():
    # 删除没有json的图片
    for imgname in os.listdir("F:\\阜康测试视频\\frame-16\\labelme\\test\\img"):
        txtfile = "F:\\阜康测试视频\\frame-16\\labelme\\test\\json\\" + imgname.split(".")[0] + ".json"
        if not os.path.exists(txtfile):
            imagefile = "F:\\阜康测试视频\\frame-16\\labelme\\test\\img\\" + imgname
            print(imagefile)
            os.remove(imagefile)
    for jsonname in os.listdir("F:\\阜康测试视频\\frame-16\\labelme\\test\\1\\json"):
        imagefile = "F:\\阜康测试视频\\frame-16\\labelme\\test\\1\\img\\" + jsonname.split(".")[0] + ".jpg"
        if not os.path.exists(imagefile):
            jsonfile = "F:\\阜康测试视频\\frame-16\\labelme\\test\\1\\json\\" + jsonname
            print(jsonfile)
            os.remove(jsonfile)

def copyDir():
    # 将标注的图片和标注文件拷贝到新文件夹
    i = 0
    num0 = 0
    num1 = 0
    num2 = 0
    num3 = 0
    num4 = 0
    imgDir = "F:\\阜康测试视频\\28-09\\标注任务分配\\dir"
    saveDir = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\"
    for root, dirs, files in os.walk(imgDir):
        for dir in dirs:
            # if (dir != "4"):
            #     continue
            imgDir1 = os.path.join(root, dir) # 一层
            for root1, dirs1, files1 in os.walk(imgDir1):
                for dir1 in dirs1:
                    imgDir2 = os.path.join(root1, dir1) # 两层
                    for root2, dirs2, files2 in os.walk(imgDir2):
                        for dir2 in dirs2:
                            imgDir3 = os.path.join(imgDir2, dir2) # 三层
                            print(imgDir3 + "\n")
                            for filename in tqdm(os.listdir(imgDir3)):
                                # if (filename.split(".")[-1] in ["json"]):
                                #     i += 1
                                #     jsonfile = os.path.join(imgDir3, filename)
                                #     imagename = filename.split(".")[0] + ".jpg"
                                #     imagefile = os.path.join(imgDir3, imagename)
                                #     if not os.path.isfile(imagefile):
                                #         continue
                                #     # print(imagefile)
                                #     newImgsaveDir = saveDir + dir + "\\image\\"
                                #     newJsonsaveDir = saveDir + dir + "\\json\\"
                                #     if not os.path.exists(newImgsaveDir):
                                #         os.makedirs(newImgsaveDir)
                                #     if not os.path.exists(newJsonsaveDir):
                                #         os.makedirs(newJsonsaveDir)
                                #     newImgfile = newImgsaveDir + imagename
                                #     newJsonfile = newJsonsaveDir + filename
                                #     copyfile(imagefile, newImgfile)
                                #     copyfile(jsonfile, newJsonfile)
                                # 统计需标注图片总数
                                if (filename.split(".")[-1] in ["jpg"]):
                                    if (dir == "0"):
                                        num0 += 1
                                    if (dir == "1"):
                                        num1 += 1
                                    if (dir == "2"):
                                        num2 += 1
                                    if (dir == "3"):
                                        num3 += 1
                                    if (dir == "4"):
                                        num4 += 1

    print("num0:" + str(num0) + "\n")
    print("num1:" + str(num1) + "\n")
    print("num2:" + str(num2) + "\n")
    print("num3:" + str(num3) + "\n")
    print("num4:" + str(num4) + "\n")

def get_hand_smoke():
    # 将28-09fukang（all文件夹）中包含手和香烟的标注拿出
    jsondir = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\0-4\\4\\json\\"
    imgdir = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\0-4\\4\\image\\"
    saveDir = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\all_2\\"
    class_key = ['hand', 'cigarette']
    for num, jsonname in enumerate(tqdm(os.listdir(jsondir))):
        json_file = jsondir + jsonname
        flag = False
        with open(json_file, 'r', encoding='utf-8') as fp:
            data = json.load(fp)  # 加载json文件
            for shapes in data['shapes']:
                label = shapes['label']
                if label in class_key:
                    flag = True
                    break

        if flag:
            source_file = json_file
            if not os.path.exists(saveDir + "json\\"):
                os.makedirs(saveDir + "json\\")
            target_file = saveDir + "json\\" + jsonname
            copyfile(source_file, target_file)
            # os.remove(source_file)

            imgfile = imgdir + jsonname.split(".")[0] + ".jpg"
            if os.path.exists(imgfile):
                source_file = imgfile
                if not os.path.exists(saveDir + "image\\"):
                    os.makedirs(saveDir + "image\\")
                target_file = saveDir + "image\\" + jsonname.split(".")[0] + ".jpg"
                copyfile(source_file, target_file)
                # os.remove(source_file)


def get_smoke():
    # 将28-09fukang_2（all_2文件夹）训练集中包含香烟的标注拿出
    jsondir = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\28-09fukang_2+DSMhand_smoke3\\val\\json\\"
    imgdir = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\28-09fukang_2+DSMhand_smoke3\\val\\image\\"
    saveDir = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\28-09fukang_2+DSMhand_smoke3-cigarette\\val\\"
    class_key = ['cigarette']
    for num, jsonname in enumerate(tqdm(os.listdir(jsondir))):
        json_file = jsondir + jsonname
        flag = False
        with open(json_file, 'r', encoding='utf-8') as fp:
            data = json.load(fp)  # 加载json文件
            for shapes in data['shapes']:
                label = shapes['label']
                if label in class_key:
                    flag = True
                    break

        if flag:
            source_file = json_file
            if not os.path.exists(saveDir + "json\\"):
                os.makedirs(saveDir + "json\\")
            target_file = saveDir + "json\\" + jsonname
            copyfile(source_file, target_file)
            # os.remove(source_file)

            imgfile = imgdir + jsonname.split(".")[0] + ".jpg"
            if os.path.exists(imgfile):
                source_file = imgfile
                if not os.path.exists(saveDir + "image\\"):
                    os.makedirs(saveDir + "image\\")
                target_file = saveDir + "image\\" + jsonname.split(".")[0] + ".jpg"
                copyfile(source_file, target_file)
                # os.remove(source_file)

def csv2json():
    # 读取csv生成json
    imgPath = "E:\\pycharm-projects\\dataset\\检测数据集\\数据堂叼烟\\sigurate"
    csvfile = "E:\\pycharm-projects\\dataset\\检测数据集\\数据堂叼烟\\sigurate_smoke.csv"
    ImgsavePath = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\28-09fukang_2+数据堂叼烟\\train\\image"
    TxtsavePath = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\28-09fukang_2+数据堂叼烟\\train\\json"
    with open(csvfile, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)
            imgfile = os.path.join(imgPath, row[0])
            img = cv2.imdecode(np.fromfile(imgfile, dtype=np.uint8), 1)
            width = img.shape[1]
            height = img.shape[0]
            target_file = os.path.join(ImgsavePath, row[0])
            if not os.path.exists(ImgsavePath):
                os.makedirs(ImgsavePath)
            # copyfile(imgfile, target_file)

            new_dict = { "version": "4.6.0", "flags": {}, "shapes": [], "imageData": None }  # label format
            new_dict["imagePath"] = row[0]
            new_dict["imageHeight"] = height
            new_dict["imageWidth"] = width
            object = {}
            object["label"] = "cigarette"
            object["group_id"] = None
            object["shape_type"] = "rectangle"
            object["flags"] = {}
            object["points"] = []
            x1y1 = [float(row[1]), float(row[2])]
            x2y2 = [float(row[3]), float(row[4])]
            object["points"].append(x1y1)
            object["points"].append(x2y2)
            new_dict["shapes"].append(object)

            json_path = os.path.join(TxtsavePath, row[0].split(".")[0])
            with open(json_path + '.json', 'a') as f:
                json.dump(new_dict, f, indent=4)

def getRandID():
    rootdir = "E:\\pycharm-projects\\dataset\\DSM_Dataset_class4_20220211_fukang\\"
    saveDir = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\all_2\\val\\"
    dict_id = {}
    for imgname in os.listdir(rootdir + "image"):
        # txtname = imgname.split(".")[0] + ".txt"
        # source_file = os.path.join("E:\\pycharm-projects\\dataset\\DSMhand&smoke1\\labels\\train", txtname)
        # target_file = os.path.join("E:\\pycharm-projects\\dataset\\DSMhand&smoke1\\labels\\val", txtname)
        # copyfile(source_file, target_file)
        # os.remove(source_file)
        ID = imgname.split("_")[1]
        if (ID in dict_id.keys()):
            dict_id[ID].append(imgname)
        else:
            dict_id[ID] = []
            dict_id[ID].append(imgname)

    # print(dict_id)
    # 随机选择1/9个ID做测试集
    val_nums = int(len(dict_id.keys()) // 9)
    val_id = random.sample(dict_id.keys(), val_nums)
    print(val_id)
    for id in tqdm(dict_id.keys()):
        if id in val_id:
            for imgname in dict_id[id]:
                source_file = rootdir + "image\\" + imgname
                if not os.path.exists(saveDir + "image\\"):
                    os.makedirs(saveDir + "image\\")
                target_file = saveDir + "image\\" + imgname
                copyfile(source_file, target_file)
                os.remove(source_file)

                txtfile = rootdir + "json\\" + imgname.split(".")[0] + ".json"
                if os.path.exists(txtfile):
                    source_file = txtfile
                    if not os.path.exists(saveDir + "json\\"):
                        os.makedirs(saveDir + "json\\")
                    target_file = saveDir + "json\\" + imgname.split(".")[0] + ".json"
                    copyfile(source_file, target_file)
                    os.remove(source_file)
        # else:
        #     for imgname in dict_id[id]:
        #         source_file = rootdir + "image\\" + imgname
        #         if not os.path.exists(saveDir + "train\\image\\"):
        #             os.makedirs(saveDir + "train\\image\\")
        #         target_file = saveDir + "train\\image\\" + imgname
        #         copyfile(source_file, target_file)
        #         # os.remove(source_file)
        #
        #         txtfile = rootdir + "json\\" + imgname.split(".")[0] + ".json"
        #         if os.path.exists(txtfile):
        #             source_file = txtfile
        #             if not os.path.exists(saveDir + "train\\json\\"):
        #                 os.makedirs(saveDir + "train\\json\\")
        #             target_file = saveDir + "train\\json\\" + imgname.split(".")[0] + ".json"
        #             copyfile(source_file, target_file)
        #             # os.remove(source_file)

def copy_plate():
    # 将香烟粘贴在阜康真实数据解决类别不均衡问题
    baseImgPath = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\28-09fukang_2_copy_plate\\train\\image"
    baseJsonPath = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\28-09fukang_2_copy_plate\\train\\json"
    saveImgPath = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\28-09fukang_2_copy_plate\\train\\image_cp"
    saveJsonPath = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\28-09fukang_2_copy_plate\\train\\json_cp"
    smokeImgPath = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\28-09fukang_2_copy_plate\\smokeArea"
    radio = 5
    if not os.path.exists(saveImgPath):
        os.makedirs(saveImgPath)
    if not os.path.exists(saveJsonPath):
        os.makedirs(saveJsonPath)
    for jsonname in tqdm(os.listdir(baseJsonPath)):
        json_file = os.path.join(baseJsonPath, jsonname)
        hasFlag = False
        with open(json_file, 'r', encoding='utf-8') as fp:
            data = json.load(fp)  # 加载json文件
            for shapes in data['shapes']:
                label = shapes['label']
                if label == "cigarette":
                    hasFlag = True
                    break
        if hasFlag:
            continue
        imgname = jsonname.split(".")[0] + ".jpg"
        baseimgfile = os.path.join(baseImgPath, imgname)
        baseImg = Image.open(baseimgfile)
        size1 = baseImg.size

        smoke_list = os.listdir(smokeImgPath)
        select_index = random.randint(0, len(smoke_list) - 1)
        smokeimgfile = os.path.join(smokeImgPath, smoke_list[select_index])
        smoke_img = Image.open(smokeimgfile)
        size2 = smoke_img.size

        center = (int(size1[0] / 2), int(size1[1] / 2))

        nums = 0
        flag = False
        while ((size1[0] < size2[0]) or (size1[1] < size2[1]) or (size2[0] < 20 or size2[1] < 20)):
            # nums += 1
            # if (nums > 100):
            #     flag = True
            #     break
            select_index = random.randint(0, len(smoke_list) - 1)
            smokeimgfile = os.path.join(smokeImgPath, smoke_list[select_index])
            smoke_img = Image.open(smokeimgfile)
            size2 = smoke_img.size


        crop_x1 = center[0] - int(size2[0] / 2)
        crop_y1 = center[1] - int(size2[1] / 2)
        crop_x2 = center[0] + (size2[0] - int(size2[0] / 2))
        crop_y2 = center[1] + (size2[1] - int(size2[1] / 2))
        baseImg.paste(smoke_img, (crop_x1, crop_y1))

        # baseImg.show()

def randDir():
    # 将数据打乱均匀分到子文件夹
    imgDir = ""
    saveDir = ""
    imglist = os.listdir(imgDir)
    nums = len(imglist)
    val_id = random.sample(imglist, 2)

# 从文件夹中随机抽取一定数量图片，分发标注任务
def randSample():
    imgDir = "F:\\阜康测试视频\\22-10\\frame-16"
    saveDir = "F:\\阜康测试视频\\22-10\\标注任务分配\\"
    imglist = os.listdir(imgDir)
    nums = len(imglist)
    people = 4
    ave_nums = math.ceil(nums / people)
    # val_id = random.sample(imglist, 2000)
    shuffle(imglist)
    for i, imagename in enumerate(tqdm(imglist)):
        sourcefile = os.path.join(imgDir, imagename)
        sonFold = str(math.floor(i / ave_nums))
        desPath = saveDir + sonFold + "\\"
        if not os.path.exists(desPath):
            os.makedirs(desPath)
        desfile = desPath + imagename
        copyfile(sourcefile, desfile)


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

def csv2json1():
    # 读取csv与已有json融合
    csvfile = "E:\\pycharm-projects\\dataset\\检测数据集\\keruisite\\keruisite_phone.csv"
    TxtsavePath = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\28-09fukang+DSMhand_smoke3\\train\\image\\D3_image\\phone_new_json"
    with open(csvfile, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            orgin_json = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\28-09fukang+DSMhand_smoke3\\train\\image\\D3_image\\" + row[0].split(".")[0] + ".json"
            flag = False
            if os.path.exists(orgin_json):
                with open(orgin_json, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)  # 加载json文件
                    for shapes in data['shapes']:
                        json_label = shapes['label']
                        if json_label == "cellphone":
                            flag = True
                            break
                    if (flag):
                        pass
                    else:
                        object = {}
                        object["label"] = "cellphone"
                        object["group_id"] = None
                        object["shape_type"] = "rectangle"
                        object["flags"] = {}
                        object["points"] = []
                        x1y1 = [float(row[1]), float(row[2])]
                        x2y2 = [float(row[3]), float(row[4])]
                        object["points"].append(x1y1)
                        object["points"].append(x2y2)
                        data["shapes"].append(object)
                newjsonfile = os.path.join(TxtsavePath, row[0].split(".")[0] + ".json")
                with open(newjsonfile, 'a') as f:
                    json.dump(data, f, indent=4)

# 合并两个json文件
def mergeJson():
    baseJsonPath1 = "F:\\阜康测试视频\\22-10\\frame-16-(0122-0210)-json"
    baseJsonPath2 = "F:\\阜康测试视频\\22-10\\返回标注文件\\all"
    TxtsavePath = "F:\\阜康测试视频\\22-10\\frame-16"
    for jsonname in tqdm(os.listdir(baseJsonPath1)):
        if jsonname.split(".")[-1] == "json":
            json_file1 = os.path.join(baseJsonPath1, jsonname)
            json_file2 = os.path.join(baseJsonPath2, jsonname)
            with open(json_file1, 'r', encoding='utf-8') as fp1:
                data1 = json.load(fp1)  # 加载json文件
                if os.path.exists(json_file2):
                    with open(json_file2, 'r', encoding='utf-8') as fp2:
                        data2 = json.load(fp2)  # 加载json文件
                        for shape in data2['shapes']:
                            data1['shapes'].append(shape)
                newjsonfile = os.path.join(TxtsavePath, jsonname)
                with open(newjsonfile, 'a') as f:
                    json.dump(data1, f, indent=4)

# 统计标注框数量
def getJsonNum():
    baseJsonPath = "F:\\阜康测试视频\\22-10\\返回标注文件\\淑茗\\0"
    nums = 0
    for jsonname in tqdm(os.listdir(baseJsonPath)):
        if jsonname.split(".")[-1] == "json":
            json_file = os.path.join(baseJsonPath, jsonname)
            with open(json_file, 'r', encoding='utf-8') as fp:
                data = json.load(fp)  # 加载json文件
                for i, shapes in enumerate(data['shapes']):
                    label = shapes['label']
                    nums += 1
    print(nums)

# 更改json文件里的错误标签
def editJson():
    baseJsonPath = "F:\\阜康测试视频\\14-21\\整理标注文件\\image\\jjc"
    TxtsavePath = "F:\\阜康测试视频\\14-21\\整理标注文件\\image\\jjc_edit"
    for jsonname in tqdm(os.listdir(baseJsonPath)):
        if jsonname.split(".")[-1] == "json":
            json_file = os.path.join(baseJsonPath, jsonname)
            hasFlag = False
            with open(json_file, 'r', encoding='utf-8') as fp:
                data = json.load(fp)  # 加载json文件
                for i, shapes in enumerate(data['shapes']):
                    label = shapes['label']
                    if label == "手机":
                        hasFlag = True
                        data['shapes'][i]['label'] = "cellphone"
            if (hasFlag):
                newjsonfile = os.path.join(TxtsavePath, jsonname)
                with open(newjsonfile, 'a') as f:
                    json.dump(data, f, indent=4)

# 随机选择1/9个ID做测试集，并保证有特定类别样本
def getRandID1():
    label_list = ['cigarette', 'cellphone']
    rootdir = "E:\\pycharm-projects\\dataset\\DSM_Dataset_class4_20220211_fukang\\train\\fukang\\28-09fukang\\"
    saveDir = "E:\\pycharm-projects\\dataset\\DSM_Dataset_class4_20220211_fukang\\val\\"
    dict_id = {}
    for jsonname in os.listdir(rootdir + "json"):
        # txtname = imgname.split(".")[0] + ".txt"
        # source_file = os.path.join("E:\\pycharm-projects\\dataset\\DSMhand&smoke1\\labels\\train", txtname)
        # target_file = os.path.join("E:\\pycharm-projects\\dataset\\DSMhand&smoke1\\labels\\val", txtname)
        # copyfile(source_file, target_file)
        # os.remove(source_file)
        imgname = jsonname.split(".")[0] + ".jpg"
        ID = jsonname.split("_")[1]
        json_file = rootdir + "json\\" + jsonname
        flag = False
        with open(json_file, 'r', encoding='utf-8') as fp:
            data = json.load(fp)  # 加载json文件
            for i, shapes in enumerate(data['shapes']):
                label = shapes['label']
                if label in label_list:
                    flag = True
        if flag:
            if (ID in dict_id.keys()):
                dict_id[ID].append(imgname)
            else:
                dict_id[ID] = []
                dict_id[ID].append(imgname)

    # print(dict_id)
    # 随机选择1/9个ID做测试集
    val_nums = int(len(dict_id.keys()) // 9)
    val_id = random.sample(dict_id.keys(), val_nums)
    print(val_id)
    for id in tqdm(dict_id.keys()):
        if id in val_id:
            for imgname in dict_id[id]:
                source_file = rootdir + "image\\" + imgname
                if not os.path.exists(saveDir + "image\\"):
                    os.makedirs(saveDir + "image\\")
                target_file = saveDir + "image\\" + imgname
                copyfile(source_file, target_file)
                os.remove(source_file)

                txtfile = rootdir + "json\\" + imgname.split(".")[0] + ".json"
                if os.path.exists(txtfile):
                    source_file = txtfile
                    if not os.path.exists(saveDir + "json\\"):
                        os.makedirs(saveDir + "json\\")
                    target_file = saveDir + "json\\" + imgname.split(".")[0] + ".json"
                    copyfile(source_file, target_file)
                    os.remove(source_file)




if __name__ == "__main__":
    # get_smoke()
    # csv2json1()
    # editJson()
    # getRandID1()
    # randSample()
    # getJsonNum()
    # randSample()
    mergeJson()