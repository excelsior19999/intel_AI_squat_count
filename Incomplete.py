import collections
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from IPython import display
from numpy.lib.stride_tricks import as_strided
from openvino.runtime import Core
from decoder import OpenPoseDecoder

import os
from collections import namedtuple
import matplotlib.pyplot as plt
import torch
from IPython.display import HTML, FileLink, display
from openvino.tools import mo

utils_file_path = Path("../utils/notebook_utils.py")
notebook_directory_path = Path(".")


sys.path.append(str(utils_file_path.parent))
sys.path.append(str(notebook_directory_path))

from notebook_utils import load_image
from model.u2net import U2NET, U2NETP


sys.path.append("../utils")
#import notebook_utils as utils

model_config = namedtuple("ModelConfig", ["name", "url", "model", "model_args"])

u2net_lite = model_config(
    name="u2net_lite",
    url="https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy",
    model=U2NETP,
    model_args=(),
)
u2net = model_config(
    name="u2net",
    url="https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
    model=U2NET,
    model_args=(3, 1),
)
u2net_human_seg = model_config(
    name="u2net_human_seg",
    url="https://drive.google.com/uc?id=1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P",
    model=U2NET,
    model_args=(3, 1),
)

u2net_model = u2net_lite
global bg_removed_result

# The filenames of the downloaded and converted models.
MODEL_DIR = "model"
model_path = Path(MODEL_DIR) / u2net_model.name / Path(u2net_model.name).with_suffix(".pth")

if not model_path.exists():
    import gdown

    os.makedirs(name=model_path.parent, exist_ok=True)
    print("Start downloading model weights file... ")
    with open(model_path, "wb") as model_file:
        gdown.download(url=u2net_model.url, output=model_file)
        print(f"Model weights have been downloaded to {model_path}")

# Set u2net_model to one of the three configurations listed above.
u2net_model = u2net_lite

# A directory where the model will be downloaded.
base_model_dir = Path("model")

# The filenames of the downloaded and converted models.
MODEL_DIR = "model"
model_path  = Path(MODEL_DIR) / u2net_model.name / Path(u2net_model.name).with_suffix(".pth")

# The name of the model from Open Model Zoo.
model_name = "human-pose-estimation-0001.xml"


# Initialize OpenVINO Runtime
ie_core = Core()
# Read the network from a file.
model = ie_core.read_model(model = model_name)
# Let the AUTO device decide where to load the model (you can use CPU, GPU or MYRIAD as well).
compiled_model = ie_core.compile_model(model=model, device_name="AUTO", config={"PERFORMANCE_HINT": "LATENCY"})

# Get the input and output names of nodes.
input_layer = compiled_model.input(0)
output_layers = compiled_model.outputs

# Get the input size.
height, width = list(input_layer.shape)[2:]


decoder = OpenPoseDecoder()


# 2D pooling in numpy (from: https://stackoverflow.com/a/54966908/1624463)
def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
    """
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    """
    # Padding
    A = np.pad(A, padding, mode="constant")

    # Window view of A
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1,
    )
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(
        A,
        shape=output_shape + kernel_size,
        strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides
    )
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling.
    if pool_mode == "max":
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == "avg":
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


# non maximum suppression
def heatmap_nms(heatmaps, pooled_heatmaps):
    return heatmaps * (heatmaps == pooled_heatmaps)


# Get poses from results.
def process_results(img, pafs, heatmaps):
    # This processing comes from
    # https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/models/open_pose.py
    pooled_heatmaps = np.array(
        [[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]]
    )
    nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)

    # Decode poses.
    poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
    output_shape = list(compiled_model.output(index=0).partial_shape)
    output_scale = img.shape[1] / output_shape[3].get_length(), img.shape[0] / output_shape[2].get_length()
    # Multiply coordinates by a scaling factor.
    poses[:, :, :2] *= output_scale
    return poses, scores


colors = ((255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85), (255, 0, 170), (85, 255, 0),
          (255, 170, 0), (0, 255, 0), (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
          (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255), (0, 170, 255))

default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
                    (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

##(15, 13): 왼쪽 어깨 관절과 왼쪽 팔꿈치 관절
#(13, 11): 왼쪽 팔꿈치 관절과 왼쪽 손목 관절
#(16, 14): 오른쪽 어깨 관절과 오른쪽 팔꿈치 관절
#(14, 12): 오른쪽 팔꿈치 관절과 오른쪽 손목 관절
##(11, 12): 왼쪽 어깨 관절과 오른쪽 어깨 관절
#(5, 11): 왼쪽 어깨 관절과 왼쪽 엉덩이 관절
#(6, 12): 오른쪽 어깨 관절과 오른쪽 엉덩이 관절
#(5, 6): 왼쪽 엉덩이 관절과 오른쪽 엉덩이 관절
#(5, 7): 왼쪽 엉덩이 관절과 왼쪽 무릎 관절
#(6, 8): 오른쪽 엉덩이 관절과 오른쪽 무릎 관절
#(7, 9): 왼쪽 무릎 관절과 왼쪽 발목 관절
#(8, 10): 오른쪽 무릎 관절과 오른쪽 발목 관절
#(1, 2): 목 관절과 코 관절
#(0, 1): 왼쪽 귀 관절과 목 관절
#(0, 2): 오른쪽 귀 관절과 목 관절
#(1, 3): 목 관절과 왼쪽 어깨 관절
#(2, 4): 목 관절과 오른쪽 어깨 관절
#(3, 5): 왼쪽 어깨 관절과 왼쪽 엉덩이 관절
#(4, 6): 오른쪽 어깨 관절과 오른쪽 엉덩이 관절

# 어깨 관절과 발목 관절 사이의 길이가 특정 범위 미만으로 가까워지면 Count+1
global shoulder # 어깨
shoulder = 0

global foot     # 발목
foot = 0

global high     # 어깨와 발목 간의 간격 길이
high = 0

global count
count = 0

global isdown       # 다시 올라갔다가 내려갔는지 판별하기 위해 존재
isdown = False

def remove_background(frame, resized_result):
    bg_removed_result = frame.copy()
    bg_removed_result[resized_result == 0] = 255

    return bg_removed_result

def draw_poses(img, poses, point_score_threshold, skeleton=default_skeleton) :
    global shoulder
    global foot
    global high
    global count
    global isdown

    if poses.size == 0:
        return img
    
    img_limbs = np.copy(img)    # img 복사가 맞음
    
    for pose in poses:
        points = pose[:, :2].astype(np.int32)   # .astype : 부호있는 32비트 정수형으로 변환해줌
        points_scores = pose[:, 2]              # 인덱스가 2미만인 값 0,1을 추출
        
        #Draw joints.
        # enumerate() : 인자로 넘어온 목록을 기준으로 인덱스와 원소를 차례로 접근하게 해주는 반복자 객체를 반환해준다
        # zip() : 각자 자료형의 개수가 동일할 때 사용, 각 n번째끼리 묶어준다.
        # for() : i : 인덱스 번호, p = points, v = points_scores
        for i, (p, v) in enumerate(zip(points,points_scores)):
            if v > point_score_threshold:
                cv2.circle(img,tuple(p), 1, colors[i], 2)
                text = f'({p[0]}, {p[1]})'
                # p[0]에 있는 자료를 문자형으로, p[1]에 있는 자료를 문자형으로 text 변수에 할당
                cv2.putText(img, text, tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

                if i==5:    # or i==6: # 한쪽 어깨를 의미. 5:왼쪽 어깨 || 6:오른쪽 어깨
                    #print("joint {}: X={}, Y={}".format(i, p[0], p[1]))
                    shoulder = f'{p[1]}'
                    #print("shoulder = " + str(shoulder))

                elif i==15:   # or i==10: # 한쪽 발목을 의미. 15:왼쪽 발목 || 10:오른쪽 발목 #
                    foot = f'{p[1]}'
                    #print("foot=" + str(foot))
                    #print("avgYval="+str(avgYval))
        high = int(foot)-int(shoulder)  # 발목과 어깨 간의 거리     (프로그램 내 숫자(단위)는 상대적)
        
        while(1):
            # # Check for the 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if high<170 and isdown == False:
                count=count+1
                print(count)
                isdown=True     # 중복 count하지 않도록 flag 역할을 한다.
            elif high<170 and isdown == True:
                continue
            elif high>170 and isdown == False:
                continue
            elif high>170 and isdown == True:
                isdown = False
                    
        
        # Draw limbs.
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                cv2.line(img_limbs,tuple(points[i]),tuple(points[j]), color=colors[j],thickness=4)
        cv2.addWeighted(img,0.4,img_limbs,0.6,0,dst=img)

        return img


def run_pose_estimation_webcam(flip=False, use_popup=False, skip_first_frames=0):
    # Load your compiled model and other necessary configurations here.
    # ...

    pafs_output_key = compiled_model.output("Mconv7_stage2_L1")
    heatmaps_output_key = compiled_model.output("Mconv7_stage2_L2")

    model_ir = mo.convert_model(
    "u2net.onnx",
    mean_values=[123.675, 116.28 , 103.53],
    scale_values=[58.395, 57.12 , 57.375],
    compress_to_fp16=True
    )

    
    # Open the webcam
    cap = cv2.VideoCapture(0)


    if use_popup:
        title = "test title"
        cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FULLSCREEN)

    processing_times = collections.deque()

    while True:
        # Read the frame from the webcama
        ret, frame = cap.read()
        if not ret:
            print("Failed to read the frame")
            break
        # If the frame is larger than full HD, reduce size to improve performance.
        scale = 1280 / max(frame.shape)
        
        resized_result = np.rint(
            cv2.resize(src=np.squeeze(frame), dsize=(frame.shape[1],frame.shape[0]))
        ).astype(np.uint8)

        
        
        
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            # 입력영상, 결과영상크기, 출력영상, x와 y 방향 스케일 비율(dsize 0일때 유효),보간법(영상 축소에 효과적)

        # Resize the image and change dimensions to fit neural network input.
        input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]
        # transpose : HWC -> CHW로 변환 OpenCV에서는 HWC, Pytorch에서는 CHW
        # H : 높이 W : 너비 C : 채널


        # Measure processing time.
        start_time = time.time()
        # Get results.
        results = compiled_model([input_img])
        stop_time = time.time()

        pafs = results[pafs_output_key]
        heatmaps = results[heatmaps_output_key]
        # Get poses from network results.
        poses, scores = process_results(frame, pafs, heatmaps)

        # Draw poses on a frame.
        frame = draw_poses(frame, poses, 0.1)

        frame=remove_background(frame, resized_result)

        processing_times.append(stop_time - start_time)
        # Use processing times from last 200 frames.
        if len(processing_times) > 200:
            processing_times.popleft()            


        if use_popup:
            cv2.imshow(title, frame)
            key = cv2.waitKey(1)
            # Escape key (27)
            if key == 27:
                break
        else:
            # Display the frame
            cv2.imshow("Webcam", frame)

            # # Check for the 'q' key to exit
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    # Release the webcam and close any open windows
    cap.release()
    cv2.destroyAllWindows()

# Run the pose estimation on webcam
run_pose_estimation_webcam(flip=True, use_popup=True)
