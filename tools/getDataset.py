# 获取烟遮挡数据的手部区域和对应烟坐标用于训练烟头检测
import os
import csv
from shutil import copyfile
import cv2
import numpy as np
import random
from tqdm import tqdm
import json

if __name__ == "__main__":
    # imgPath = "E:\\pycharm-projects\\dataset\\检测数据集\\keruisite\\sigurate"
    # csvfile = "E:\\pycharm-projects\\dataset\\检测数据集\\keruisite\\sigurate_smoke.csv"
    # ImgsavePath = "E:\\pycharm-projects\\dataset\\DSMhand&smoke2\\temp\\keruisite\\sigurate"
    # TxtsavePath = "E:\\pycharm-projects\\dataset\\DSMhand&smoke2\\temp\\keruisite\\sigurate"
    # with open(csvfile, 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         print(row)
    #         imgfile = os.path.join(imgPath, row[0])
    #         img = cv2.imdecode(np.fromfile(imgfile, dtype=np.uint8), 1)
    #         width = img.shape[1]
    #         height = img.shape[0]
    #         target_file = os.path.join(ImgsavePath, row[0])
    #         if not os.path.exists(ImgsavePath):
    #             os.makedirs(ImgsavePath)
    #         copyfile(imgfile, target_file)
    #         txtfile = os.path.join(TxtsavePath, row[0].split(".")[0] + ".txt")
    #         with open(txtfile, "a") as f1:
    #             # hand 0  smoke 1
    #             label = 1
    #             object_x_center = (int(row[1]) + int(row[3])) / 2.0 / float(width)
    #             object_y_center = (int(row[2]) + int(row[4])) / 2.0 / float(height)
    #             object_width = (int(row[3]) - int(row[1])) / float(width)
    #             object_height = (int(row[4]) - int(row[2])) / float(height)
    #             f1.write(str(label) + " " + str(object_x_center) + " " + str(object_y_center) + " " + str(object_width) + " " + str(object_height) + "\n")

    rootdir = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\all_2\\train\\"
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

    # # 删除没有json的图片
    # for imgname in os.listdir("F:\\阜康测试视频\\frame-16\\labelme\\test\\img"):
    #     txtfile = "F:\\阜康测试视频\\frame-16\\labelme\\test\\json\\" + imgname.split(".")[0] + ".json"
    #     if not os.path.exists(txtfile):
    #         imagefile = "F:\\阜康测试视频\\frame-16\\labelme\\test\\img\\" + imgname
    #         print(imagefile)
    #         os.remove(imagefile)

    # for jsonname in os.listdir("F:\\阜康测试视频\\frame-16\\labelme\\test\\1\\json"):
    #     imagefile = "F:\\阜康测试视频\\frame-16\\labelme\\test\\1\\img\\" + jsonname.split(".")[0] + ".jpg"
    #     if not os.path.exists(imagefile):
    #         jsonfile = "F:\\阜康测试视频\\frame-16\\labelme\\test\\1\\json\\" + jsonname
    #         print(jsonfile)
    #         os.remove(jsonfile)

    # 将标注的图片和标注文件拷贝到新文件夹
    # i = 0
    # num0 = 0
    # num1 = 0
    # num2 = 0
    # num3 = 0
    # num4 = 0
    # imgDir = "F:\\阜康测试视频\\28-09\\标注任务分配\\dir"
    # saveDir = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\"
    # for root, dirs, files in os.walk(imgDir):
    #     for dir in dirs:
    #         # if (dir != "4"):
    #         #     continue
    #         imgDir1 = os.path.join(root, dir) # 一层
    #         for root1, dirs1, files1 in os.walk(imgDir1):
    #             for dir1 in dirs1:
    #                 imgDir2 = os.path.join(root1, dir1) # 两层
    #                 for root2, dirs2, files2 in os.walk(imgDir2):
    #                     for dir2 in dirs2:
    #                         imgDir3 = os.path.join(imgDir2, dir2) # 三层
    #                         print(imgDir3 + "\n")
    #                         for filename in tqdm(os.listdir(imgDir3)):
    #                             # if (filename.split(".")[-1] in ["json"]):
    #                             #     i += 1
    #                             #     jsonfile = os.path.join(imgDir3, filename)
    #                             #     imagename = filename.split(".")[0] + ".jpg"
    #                             #     imagefile = os.path.join(imgDir3, imagename)
    #                             #     if not os.path.isfile(imagefile):
    #                             #         continue
    #                             #     # print(imagefile)
    #                             #     newImgsaveDir = saveDir + dir + "\\image\\"
    #                             #     newJsonsaveDir = saveDir + dir + "\\json\\"
    #                             #     if not os.path.exists(newImgsaveDir):
    #                             #         os.makedirs(newImgsaveDir)
    #                             #     if not os.path.exists(newJsonsaveDir):
    #                             #         os.makedirs(newJsonsaveDir)
    #                             #     newImgfile = newImgsaveDir + imagename
    #                             #     newJsonfile = newJsonsaveDir + filename
    #                             #     copyfile(imagefile, newImgfile)
    #                             #     copyfile(jsonfile, newJsonfile)
    #                             # 统计需标注图片总数
    #                             if (filename.split(".")[-1] in ["jpg"]):
    #                                 if (dir == "0"):
    #                                     num0 += 1
    #                                 if (dir == "1"):
    #                                     num1 += 1
    #                                 if (dir == "2"):
    #                                     num2 += 1
    #                                 if (dir == "3"):
    #                                     num3 += 1
    #                                 if (dir == "4"):
    #                                     num4 += 1
    #
    # print("num0:" + str(num0) + "\n")
    # print("num1:" + str(num1) + "\n")
    # print("num2:" + str(num2) + "\n")
    # print("num3:" + str(num3) + "\n")
    # print("num4:" + str(num4) + "\n")


    # # 将28-09fukang（all文件夹）中包含手和香烟的标注拿出
    # jsondir = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\0-4\\4\\json\\"
    # imgdir = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\0-4\\4\\image\\"
    # saveDir = "F:\\阜康测试视频\\28-09\\整理返回标注数据\\all_2\\"
    # class_key = ['hand', 'cigarette']
    # for num, jsonname in enumerate(tqdm(os.listdir(jsondir))):
    #     json_file = jsondir + jsonname
    #     flag = False
    #     with open(json_file, 'r', encoding='utf-8') as fp:
    #         data = json.load(fp)  # 加载json文件
    #         for shapes in data['shapes']:
    #             label = shapes['label']
    #             if label in class_key:
    #                 flag = True
    #                 break
    #
    #     if flag:
    #         source_file = json_file
    #         if not os.path.exists(saveDir + "json\\"):
    #             os.makedirs(saveDir + "json\\")
    #         target_file = saveDir + "json\\" + jsonname
    #         copyfile(source_file, target_file)
    #         # os.remove(source_file)
    #
    #         imgfile = imgdir + jsonname.split(".")[0] + ".jpg"
    #         if os.path.exists(imgfile):
    #             source_file = imgfile
    #             if not os.path.exists(saveDir + "image\\"):
    #                 os.makedirs(saveDir + "image\\")
    #             target_file = saveDir + "image\\" + jsonname.split(".")[0] + ".jpg"
    #             copyfile(source_file, target_file)
    #             # os.remove(source_file)





