import os
import random
from shutil import copyfile

if __name__ == "__main__":
    # 遍历特定文件夹
    imgDir = "F:\\阜康测试视频\\frame-16\\labelme\\train"
    i = 0
    for root, dirs, files in os.walk(imgDir):
        for dir in dirs:
            imgDir1 = os.path.join(root, dir)
            for root1, dirs1, files1 in os.walk(imgDir1):
                for dir1 in dirs1:
                    imgDir2 = os.path.join(root1, dir1)
                    for root2, dirs2, files2 in os.walk(imgDir2):
                        for dir2 in dirs2:
                            imgDir3 = os.path.join(imgDir2, dir2)
                            imgList = []
                            jsonList = []
                            for filename in os.listdir(imgDir3):
                                i += 1
                                if (filename.split(".")[1] == "jpg"):
                                    imgList.append(filename)
                                elif (filename.split(".")[1] == "json"):
                                    jsonList.append(filename)
                            # 随机从每个文件夹选取2张图片
                            if (len(imgList) > 2):
                                val_id = random.sample(imgList, 2)
                                for lastImgname in val_id:
                                    source_file = os.path.join(imgDir3, lastImgname)
                                    print(source_file)
                                    target_file = "F:\\阜康测试视频\\frame-16\\labelme\\train_labelme\\" + lastImgname
                                    copyfile(source_file, target_file)
                                    # 如果存在对应预标注文件也拷贝
                                    jsonname = lastImgname.split(".")[0] + ".json"
                                    json_file = os.path.join(imgDir3, jsonname)
                                    if (os.path.exists(json_file)):
                                        source_file1 = os.path.join(imgDir3, jsonname)
                                        target_file1 = "F:\\阜康测试视频\\frame-16\\labelme\\train_labelme\\" + jsonname
                                        copyfile(source_file1, target_file1)