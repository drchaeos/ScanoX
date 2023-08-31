## 필요한 모듈 import

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import csv

import torch
from typing import Tuple, List, Sequence, Callable, Dict

from detectron2.structures import BoxMode
from detectron2.engine import HookBase
from detectron2.utils.events import get_event_storage
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

## parameter 설정
dataname = 'HKAA'
column_path = './columns/HKAA.csv'             # column 파일 들어간 directory
test_dir = './dataset/HKAA2/test_imgs'         # test image 들어간 directory
model_dir = './out/HKAA2'                      # model.pth 들어간 directory
output_dir = './out/HKAA2'                     # output파일들 저장될 directory
num_keypoints = 3
keypoint_names = {0: 'femoral_head_center', 1: 'tibia_plateau_center', 2: 'talus_center'}
edges = [(0, 1), (1, 2)]

## 함수 정의
def calculate_HKAA(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(b[1] - a[1], b[0] - a[0]) - np.arctan2(b[1] - c[1], b[0] - c[0])
    angle = radians*180.0/np.pi

    # if abs(angle) > 180.0:
    #     angle = 360-abs(angle)

    return angle

def draw_keypoints(image, keypoints,
                   edges: List[Tuple[int, int]] = None,
                   keypoint_names: Dict[int, str] = None,
                   boxes: bool = True) -> None:

    keypoints = keypoints.astype(np.int64)
    keypoints_ = keypoints.copy()
    np.random.seed(42)
    colors = {k: tuple(map(int, np.random.randint(0, 255, 3))) for k in range(num_keypoints)}
    if len(keypoints_) == (2 * num_keypoints):
        keypoints_ = [[keypoints_[i], keypoints_[i + 1]] for i in range(0, len(keypoints_), 2)]

    assert isinstance(image, np.ndarray), "image argument does not numpy array."
    image_ = np.copy(image)
    for i, keypoint in enumerate(keypoints_):
        cv2.circle(
            image_,
            tuple(keypoint),
            3, colors.get(i), thickness=3, lineType=cv2.FILLED)

        if keypoint_names is not None:
            cv2.putText(
                image_,
                f'{i}: {keypoint_names[i]}',
                tuple(keypoint),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if edges is not None:
        for i, edge in enumerate(edges):
            cv2.line(
                image_,
                tuple(keypoints_[edge[0]]),
                tuple(keypoints_[edge[1]]),
                colors.get(edge[0]), 3, lineType=cv2.LINE_AA)
    if boxes:
        x1, y1 = min(np.array(keypoints_)[:, 0]), min(np.array(keypoints_)[:, 1])
        x2, y2 = max(np.array(keypoints_)[:, 0]), max(np.array(keypoints_)[:, 1])
        cv2.rectangle(image_, (x1, y1), (x2, y2), (255, 100, 91), thickness=3)

    h, w, c = image.shape

    global right_angle, left_angle
    if keypoints_[1][0] < (w / 2):
        right_angle = calculate_HKAA(keypoints_[0], keypoints_[1], keypoints_[2]) - 180
        right_angle = format(right_angle, ".3f")
    if keypoints_[1][0] > (w / 2):
        left_angle = 180 - calculate_HKAA(keypoints_[0], keypoints_[1], keypoints_[2])
        left_angle = format(left_angle, ".3f")

    return image_


def save_samples(dst_path, image_path, csv_path, mode="random", size=None, index=None):
    df = pd.read_csv(csv_path)
    # csv 파일로 저장
    output_file = open(f'{output_dir}/HKAA_result.csv', 'w', newline='')
    f = csv.writer(output_file)
    # csv 파일에 header 추가
    f.writerow(["image", "Rt HKAA", "Lt HKAA"])

    if mode == "random":
        assert size is not None, "mode argument is random, but size argument is not given."
        choice_idx = np.random.choice(len(df), size=size, replace=False)
    if mode == "choice":
        assert index is not None, "mode argument is choice, but index argument is not given."
        choice_idx = index

    global right_angle, left_angle

    for idx in choice_idx:
        image_name = df.iloc[idx, 0]
        keypoints = df.iloc[idx, 1:]
        image = cv2.imread(os.path.join(image_path, image_name), cv2.IMREAD_COLOR)
        if idx == 0:
            combined = draw_keypoints(image, keypoints, edges, keypoint_names, boxes=False)
            if image_name != df.iloc[idx + 1, 0]:
                cv2.imwrite(os.path.join(dst_path, "result_" + image_name), combined)
                f.writerow([image_name, right_angle, left_angle])
                right_angle = "NaN"
                left_angle = "NaN"

        if 0 < idx < (len(choice_idx)-1):
            if image_name == df.iloc[idx + 1, 0]:
                if image_name != df.iloc[idx - 1, 0]:
                    combined = draw_keypoints(image, keypoints, edges, keypoint_names, boxes=False)
                if image_name == df.iloc[idx - 1, 0]:
                    combined = draw_keypoints(combined, keypoints, edges, keypoint_names, boxes=False)

            if image_name != df.iloc[idx + 1, 0]:
                if image_name != df.iloc[idx - 1, 0]:
                    combined = draw_keypoints(image, keypoints, edges, keypoint_names, boxes=False)
                    cv2.imwrite(os.path.join(dst_path, "result_" + image_name), combined)
                    f.writerow([image_name, right_angle, left_angle])
                    right_angle = "NaN"
                    left_angle = "NaN"

                if image_name == df.iloc[idx - 1, 0]:
                    combined = draw_keypoints(combined, keypoints, edges, keypoint_names, boxes=False)
                    cv2.imwrite(os.path.join(dst_path, "result_" + image_name), combined)
                    f.writerow([image_name, right_angle, left_angle])
                    right_angle = "NaN"
                    left_angle = "NaN"

        if idx == (len(choice_idx)-1):
            if image_name != df.iloc[idx - 1, 0]:
                combined = draw_keypoints(image, keypoints, edges, keypoint_names, boxes=False)
                cv2.imwrite(os.path.join(dst_path, "result_" + image_name), combined)
                f.writerow([image_name, right_angle, left_angle])

            if image_name == df.iloc[idx - 1, 0]:
                combined = draw_keypoints(combined, keypoints, edges, keypoint_names, boxes=False)
                cv2.imwrite(os.path.join(dst_path, "result_" + image_name), combined)
                f.writerow([image_name, right_angle, left_angle])

## inference
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 0  # On Windows environment, this value must be 0.
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = num_keypoints
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((num_keypoints, 1), dtype=float).tolist()
cfg.OUTPUT_DIR = output_dir

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_final.pth")  # 학습된 모델 들어가 있는 곳
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # custom testing threshold
predictor = DefaultPredictor(cfg)

test_list = os.listdir(test_dir)
test_list.sort()
except_list = []

files = []
preds = []
for file in tqdm(test_list):
    filepath = os.path.join(test_dir, file)
    im = cv2.imread(filepath)
    outputs = predictor(im)
    for i in range(len(outputs["instances"])):
        pred_keypoints = outputs["instances"][i].to("cpu").get("pred_keypoints").numpy()
        files.append(file)
        pred = []
        try:
            for out in pred_keypoints[0]:
                pred.extend([float(e) for e in out[:2]])
        except IndexError:
            pred.extend([0] * (2 * num_keypoints))
            except_list.append(filepath)
        preds.append(pred)

df_sub = pd.read_csv(column_path)
df = pd.DataFrame(columns=df_sub.columns)
df["image"] = files
df.iloc[:, 1:] = preds

df.to_csv(os.path.join(cfg.OUTPUT_DIR, f"{dataname}_keypoints.csv"), index=False)
if except_list:
    print(
        "The following images are not detected keypoints. The row corresponding that images names would be filled with 0 value."
    )
    print(*except_list)


save_samples(cfg.OUTPUT_DIR, test_dir, os.path.join(cfg.OUTPUT_DIR, f"{dataname}_keypoints.csv"), mode="choice", size=5, index=range(len(files)))





