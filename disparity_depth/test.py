# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 21:04:06 2020

@author: gkm0120
"""
import numpy as np
import cv2


# 使用普通摄像头进行深度估计
def update(val=0):
    # 为“aloe”图像对调整视差范围
    stereo.setBlockSize(cv2.getTrackbarPos('window_size', 'disparity'))
    stereo.setUniquenessRatio(cv2.getTrackbarPos('uniquenessRatio', 'disparity'))
    stereo.setSpeckleWindowSize(cv2.getTrackbarPos('speckleWindowSize', 'disparity'))
    stereo.setSpeckleRange(cv2.getTrackbarPos('speckleRange', 'disparity'))
    stereo.setDisp12MaxDiff(cv2.getTrackbarPos('disp12MaxDiff', 'disparity'))

    print('computing disparity')
    # 将整数的视差值转化为浮点数，便于更好的表示
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    cv2.imshow('left', imgL)
    cv2.imshow('right',imgR)
    # 归一化处理
    cv2.imshow('disparity', (disp - min_disp) / num_disp)


if __name__ == "__main__":
    window_size = 5
    min_disp = 16   # 最小视差值
    num_disp = 192 - min_disp   # 视差范围
    blockSize = window_size # 每个像素周围的块大小
    uniquenessRatio = 1 # 唯一性比率，判断是否存在唯一的匹配，值越小，唯一性匹配的要求越高
    speckleRange = 3    # 斑点范围
    speckleWindowSize = 3   # 斑点窗口大小
    disp12MaxDiff = 200 # 最大视差差
    P1 = 600    # 控制低纹理区域的平滑程度
    P2 = 2400   # 控制高纹理区域的平滑程度
    # imgL = cv2.imread('image/teddy-png-2/teddy/im2.png')
    # imgR = cv2.imread('image/teddy-png-2/teddy/im6.png')
    imgL = cv2.imread('image/left_eye.png')
    imgR = cv2.imread('image/right_eye.png')

    cv2.namedWindow('disparity')
    cv2.createTrackbar('speckleRange', 'disparity', speckleRange, 50, update)
    cv2.createTrackbar('window_size', 'disparity', window_size, 21, update)
    cv2.createTrackbar('speckleWindowSize', 'disparity', speckleWindowSize, 200, update)
    cv2.createTrackbar('uniquenessRatio', 'disparity', uniquenessRatio, 50, update)
    cv2.createTrackbar('disp12MaxDiff', 'disparity', disp12MaxDiff, 250, update)
    # 创建一个StereoSGBM实例
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        uniquenessRatio=uniquenessRatio,
        speckleRange=speckleRange,
        speckleWindowSize=speckleWindowSize,
        disp12MaxDiff=disp12MaxDiff,
        P1=P1,
        P2=P2
    )
    update()
    cv2.waitKey()
