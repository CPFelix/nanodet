import argparse
import os
import time

import cv2
import torch

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.path import mkdir

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
video_ext = ["mp4", "mov", "avi", "mkv"]

import json

# cuda同步时间
def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("--config", help="model config file path")
    parser.add_argument("--model", help="model file path")
    parser.add_argument("--path", default="./demo", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    args = parser.parse_args()
    return args


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        # 重新保存模型得以使用pytorch1.4加载
        torch.save(ckpt, 'workspace/nanodet-plus-m_MobileNetV2_320X192_DSM_Dataset_class4_20220211_fukang/model_best/nanodet_model_best_resave.pth', _use_new_zipfile_serialization=False)

        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        result_img = self.model.head.show_result(
            meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=False
        )
        print("viz time: {:.3f}s".format(time.time() - time1))
        return result_img


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names

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

# 做补充预标注(脸、手、手机)
def editJson(image_name, dets, score_thresh):
    all_box = []
    for label in dets:
        for bbox in dets[label]:
            score = bbox[-1]
            if score > score_thresh:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                all_box.append([label, x0, y0, x1, y1, score])
    all_box.sort(key=lambda v: v[5])
    # jsonfile = image_name.replace("D3_image", "D3_json").replace(".jpg", ".json")
    # newjsonfile = jsonfile.replace("D3_json", "new_D3_json")
    jsonfile = image_name.replace("1421_image", "1421_json").replace(".jpg", ".json")
    newjsonfile = jsonfile.replace("1421_json", "1421_new_json")
    if os.path.exists(jsonfile):
        with open(jsonfile, 'r', encoding='utf-8') as fp:
            data = json.load(fp)  # 加载json文件
            for box in all_box:
                # 是否跟json文件中标注框重复标志位
                flag = False
                label, x0, y0, x1, y1, score = box
                if label == 0:
                    label_name = "face"
                elif label == 1:
                    label_name = "hand"
                elif label == 2:
                    continue
                    label_name = "cigarette"
                elif label == 3:
                    continue
                    label_name = "cellphone"

                for shapes in data['shapes']:
                    json_label = shapes['label']
                    boxA = [shapes['points'][0][0], shapes['points'][0][1], shapes['points'][1][0], shapes['points'][1][1]]
                    boxB = [x0, y0, x1, y1]
                    if (label_name == json_label):
                        iou = bb_intersection_over_union(boxA, boxB)
                        if (iou > 0.5):
                            flag = True
                            break
                if (flag):
                    continue
                else:
                    object = {}
                    object["label"] = label_name
                    object["group_id"] = None
                    object["shape_type"] = "rectangle"
                    object["flags"] = {}
                    object["points"] = []
                    x1y1 = [float(x0), float(y0)]
                    x2y2 = [float(x1), float(y1)]
                    object["points"].append(x1y1)
                    object["points"].append(x2y2)
                    data["shapes"].append(object)
            with open(newjsonfile, 'a') as f:
                json.dump(data, f, indent=4)
    else:
        new_dict = {"version": "4.6.0", "flags": {}, "shapes": [], "imageData": None}  # label format
        new_dict["imagePath"] = image_name.split("/")[-1]
        img = cv2.imread(image_name)
        new_dict["imageHeight"] = img.shape[0]
        new_dict["imageWidth"] = img.shape[1]
        for box in all_box:
            label, x0, y0, x1, y1, score = box
            object = {}
            if label == 0:
                object["label"] = "face"
            elif label == 1:
                object["label"] = "hand"
            elif label == 2:
                object["label"] = "cigarette"
            elif label == 3:
                object["label"] = "cellphone"
            object["group_id"] = None
            object["shape_type"] = "rectangle"
            object["flags"] = {}
            object["points"] = []
            x1y1 = [float(x0), float(y0)]
            x2y2 = [float(x1), float(y1)]
            object["points"].append(x1y1)
            object["points"].append(x2y2)
            new_dict["shapes"].append(object)
        with open(newjsonfile, 'a') as f:
            json.dump(new_dict, f, indent=4)



def main():
    args = parse_args()
    local_rank = 0
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False

    load_config(cfg, args.config)
    logger = Logger(local_rank, use_tensorboard=False)
    predictor = Predictor(cfg, args.model, logger, device="cuda:0")
    # predictor = Predictor(cfg, args.model, logger, device="cpu")
    logger.log('Press "Esc", "q" or "Q" to exit.')
    current_time = time.localtime()
    if args.demo == "image":
        # 计算推理时间
        totalTime = 0.0
        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()
        for image_name in files:
            # 计算推理时间
            t1 = time_sync()
            meta, res = predictor.inference(image_name)
            # 计算推理时间
            t2 = time_sync()
            totalTime += (t2 - t1) * 1000.0
            result_image = predictor.visualize(res[0], meta, cfg.class_names, 0.35)

            # 做补充预标注(脸、手、手机)
            # editJson(image_name, res[0], 0.5)

            if args.save_result:
                save_folder = os.path.join(
                    cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                )
                mkdir(local_rank, save_folder)
                save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                cv2.imwrite(save_file_name, result_image)
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        # 打印总时间和平均时间
        print(f'totalTime: ({totalTime:.3f}ms)')
        aveTime = totalTime / len(files)
        print(f'aveTime: ({aveTime:.3f}ms)')
    elif args.demo == "video" or args.demo == "webcam":
        cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        save_folder = os.path.join(
            cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        mkdir(local_rank, save_folder)
        save_path = (
            os.path.join(save_folder, args.path.split("/")[-1])
            if args.demo == "video"
            else os.path.join(save_folder, "camera.mp4")
        )
        print(f"save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                meta, res = predictor.inference(frame)
                result_frame = predictor.visualize(res[0], meta, cfg.class_names, 0.35)
                if args.save_result:
                    vid_writer.write(result_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break


if __name__ == "__main__":
    main()
