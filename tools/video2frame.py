import numpy as np
import cv2
import os
import sys
from PIL import Image
from tqdm import tqdm

def cut(video_file, target_dir):
    videoname = video_file.split("\\")[-1].split(".")[0]
    cap = cv2.VideoCapture(video_file)  # 获取到一个视频
    isOpened = cap.isOpened  # 判断是否打开

    # 为单张视频，以视频名称所谓文件名，创建文件夹
    # temp = os.path.split(video_file)[-1]
    # dir_name = temp.split('.')[0]
    dir_name = ""

    single_pic_store_dir = os.path.join(target_dir, dir_name)
    if not os.path.exists(single_pic_store_dir):
        os.makedirs(single_pic_store_dir)


    i = 0
    while isOpened:
        i += 1
        # print(i)
        (flag, frame) = cap.read()  # 读取一张图像

        # 隔帧抽取
        rate = 16  # 4帧抽一张
        if (i % rate != 0):
            continue

        fileName1 = videoname + "_" + str(i) + ".jpg"
        data = video_file.split("\\")[3].split("-")[1]
        fileName = data + "_" + fileName1
        # fileName = str(i) + ".jpg"
        if (flag == True):
            # 以下三行 进行 旋转
            #frame = np.rot90(frame, -1)
            #print(fileName)
            # 设置保存路径
            save_path = os.path.join(single_pic_store_dir, fileName)
            #print(save_path)
            # cv2.imshow("", frame)
            # cv2.waitKey(0)
            # cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imencode('.jpg', frame)[1].tofile(save_path)  # 存储失败
            # image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # image.show()
            # cv2.waitKey(0)
            # image.save(save_path)
            #print(res)
        else:
            break

    return single_pic_store_dir

if __name__ == '__main__':
    # video_file = 'F:\\阜康测试视频\\video\\test\\videos-20220103\\抽烟报警(主驾)\\新B27700_1585.mp4'
    # savePath = video_file.split(".")[0].replace("\\video\\", "\\frame\\") + "\\"
    # cut(video_file, savePath)
    videoDir = "F:\\阜康公安--试用告警视频\\28-09"
    saveDir = "F:\\阜康测试视频\\28-09\\"
    i = 0
    do_list = ["抽烟报警(副驾)", "抽烟报警(后排)", "抽烟报警(主驾)", "接打电话报警(主驾)"]
    for root, dirs, files in os.walk(videoDir):
        for dir in dirs:
            videoDir1 = os.path.join(root, dir)
            for root1, dirs1, files1 in os.walk(videoDir1):
                for dir1 in dirs1:
                    if dir1 in do_list:
                        videoDir2 = os.path.join(root1, dir1)
                        print(videoDir2)
                        for file in tqdm(os.listdir(videoDir2)):
                            i += 1
                            videofile = os.path.join(videoDir2, file)
                            # print(videofile)
                            LPR = file.split("_")[0]
                            # savePath = videofile.split(".")[0].replace("\\video\\", "\\frame-16\\") + "\\"
                            savePath = saveDir + dir1 + "\\" + LPR + "\\"
                            # print(savePath)
                            cut(videofile, savePath)
    print(i)


