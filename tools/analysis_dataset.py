import os
from tqdm import tqdm
import json



# 统计标注框的尺寸分布
def analysis_json():
    baseJsonPath = "E:\\pycharm-projects\\dataset\\DSM_Dataset_class4_20220211_fukang\\val\\json"
    dict_class = {}
    for jsonname in tqdm(os.listdir(baseJsonPath)):
        if jsonname.split(".")[-1] == "json":
            json_file = os.path.join(baseJsonPath, jsonname)
            with open(json_file, 'r', encoding='utf-8') as fp:
                data = json.load(fp)  # 加载json文件
                for i, shapes in enumerate(data['shapes']):
                    label = shapes['label']
                    if label not in dict_class.keys():
                        dict_class[label] = [0, 0, 0]  # ws, hs, nums
                    w = shapes['points'][1][0] - shapes['points'][0][0]
                    h = shapes['points'][1][1] - shapes['points'][0][1]
                    dict_class[label][0] += w
                    dict_class[label][1] += h
                    dict_class[label][2] += 1
    # print(dict_class)
    for k, v in dict_class.items():
        print("%s:\nave_W:%d ave_H:%d nums:%d"%(k, v[0]/v[2], v[1]/v[2], v[2]))



if __name__ == "__main__":
    analysis_json()