# Import required libraries
# 导入所需的库
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
import clip
from datetime import datetime


# Load models (YOLO12 and CLIP)
# 加载模型（YOLO12和CLIP）
def load_models(model_yolo_path="yolo12n.pt", model_clip_name="ViT-B/32"):

    print("Loading yolo12n model...")
    print("正在加载yolo12n模型...")
    model_yolo = YOLO(model_yolo_path)

    print("Loading CLIP model...")
    print("正在加载CLIP模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_clip, preprocess = clip.load(model_clip_name, device=device)

    return model_yolo, model_clip, preprocess, device
