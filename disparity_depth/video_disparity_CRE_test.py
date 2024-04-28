import os
import threading
from time import sleep

import cv2
import numpy as np
import open3d as o3d
import pyzed.sl as sl

import compute_utils
from camera_utils import get_camera_intrinsics, get_image_size
from nets import Model
import torch
import torch.nn.functional as F

device = 'cuda'
# model_path = "models/crestereo_eth3d.pth"
model_path = "../models/crestereo_eth3d.pth"
model = Model(max_disp=256, mixed_precision=False, test_mode=True)
model.load_state_dict(torch.load(model_path), strict=True)
model.to(device)
model.eval()

zed = sl.Camera()
init = sl.InitParameters()
init.depth_mode = sl.DEPTH_MODE.ULTRA
# init.camera_resolution = sl.RESOLUTION.HD720
init.camera_resolution = sl.RESOLUTION.VGA

status = zed.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    print(repr(status))
    camera_status = False
else:
    print("Camera open successfully!")
    camera_status = True

runtime_params = sl.RuntimeParameters()

left = sl.Mat()
right = sl.Mat()

# 获取摄像头内参矩阵，焦距和基线距离
camera_intrinsics, focal_left_x, baseline_mm = get_camera_intrinsics(zed)
image_size = get_image_size(zed)

while camera_status:
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        # 获取左,右图像
        zed.retrieve_image(left, sl.VIEW.LEFT)
        zed.retrieve_image(right, sl.VIEW.RIGHT)

        left_data = left.get_data()
        right_data = right.get_data()

        # BGRA convert to BGR
        left_data = cv2.cvtColor(left_data, cv2.COLOR_BGRA2BGR)
        right_data = cv2.cvtColor(right_data, cv2.COLOR_BGRA2BGR)

        disp = compute_utils.compute_disparity_CRE(left_data, right_data)
        depth = compute_utils.compute_depth(disp, baseline_mm, focal_left_x)

        # 可视化视差图
        disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
        disp_vis = disp_vis.astype("uint8")
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

        cv2.imshow("disp", disp_vis)
        cv2.imshow("depth", depth)

        key = cv2.waitKey(10)
        if key == ord('q') or key == ord('Q'):
            exit_program = True
            print("Exit camera thread: ", exit_program)
            break
