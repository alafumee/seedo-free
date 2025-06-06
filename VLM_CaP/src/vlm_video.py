from IPython.display import display, Image, Audio

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
import numpy as np
from openai import OpenAI
import os
import requests

def extract_frames(video_path):
    # 使用 OpenCV 从视频文件中提取帧
    video = cv2.VideoCapture(video_path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    print(len(base64Frames), "frames read.")
    
    # 展示帧，用于调试
    # display_handle = display(None, display_id=True)
    # for img in base64Frames:
    #     display_handle.update(Image(data=base64.b64decode(img.encode("utf-8"))))
        # time.sleep(0.025)  # 如果需要以动画形式展示每帧，可以取消此行注释

    return base64Frames  # 返回包含所有帧的 base64 编码的列表

def extract_frame_list(frames_list):
    """
    将图像帧列表转换为 Base64 编码的列表。
    
    :param frames_list: 图像帧列表，每一帧是一个 NumPy 数组（从 OpenCV 读取的图像）。
    :return: 包含所有帧的 Base64 编码字符串列表。
    """
    base64Frames = []

    for frame in frames_list:
        if frame is None:
            continue
        
        # 编码帧为 JPEG 格式
        _, buffer = cv2.imencode(".jpg", frame)
        
        # 将 JPEG 编码的帧转换为 Base64 字符串
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    print(len(base64Frames), "frames processed.")

    return base64Frames  # 返回包含所有帧的 Base64 编码的列表

def base64_to_cv2(base64_str):
    """
    将 Base64 编码的字符串转换为 OpenCV 图像。
    
    :param base64_str: Base64 编码的图像字符串。
    :return: OpenCV 图像（NumPy 数组）。
    """
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img